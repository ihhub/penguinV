#include <algorithm>
#include "image_exception.h"
#include "thread_pool.h"

namespace Thread_Pool
{
    AbstractTaskProvider::AbstractTaskProvider()
        : _taskCount         ( 0 )
        , _givenTaskCount    ( 0 )
        , _completedTaskCount( 0 )
        , _running           ( false )
        , _exceptionRaised   ( false )
    {
    }

    AbstractTaskProvider::AbstractTaskProvider( const AbstractTaskProvider & )
        : _taskCount         ( 0 )
        , _givenTaskCount    ( 0 )
        , _completedTaskCount( 0 )
        , _running           ( false )
        , _exceptionRaised   ( false )
    {
    }

    AbstractTaskProvider::~AbstractTaskProvider()
    {
        _wait();
    }

    AbstractTaskProvider & AbstractTaskProvider::operator=( const AbstractTaskProvider & )
    {
        return (*this);
    }

    void AbstractTaskProvider::_taskRun( bool skip )
    {
        size_t taskId = _givenTaskCount++;

        if( taskId < _taskCount ) {
            if( !skip ) {
                try {
                    _task( taskId );
                }
                catch( ... ) {
                    // here should be some logging code stating about an exception
                    // or add your code to feedback about an exception
                    _exceptionRaised = true;
                }
            }

            size_t completedTasks = (++_completedTaskCount);

            if( completedTasks == _taskCount ) {
                _completion.lock();

                _running = false;
                _waiting.notify_one();

                _completion.unlock();
            }
        }
    }

    bool AbstractTaskProvider::_wait()
    {
        std::unique_lock < std::mutex > _mutexLock( _completion );
        _waiting.wait( _mutexLock, [&] { return !_running; } );

        bool noException = !_exceptionRaised;
        _exceptionRaised = false;

        return noException;
    }

    bool AbstractTaskProvider::_ready() const
    {
        return !_running;
    }


    TaskProvider::TaskProvider()
        : _threadPool ( nullptr )
    {
    }

    TaskProvider::TaskProvider( ThreadPool * pool )
        : _threadPool ( pool )
    {
    }

    TaskProvider::TaskProvider( const TaskProvider & provider )
        : _threadPool ( provider._threadPool )
    {
    }

    TaskProvider::~TaskProvider()
    {
        if( _threadPool != nullptr )
            _threadPool->remove( this );
    }

    TaskProvider & TaskProvider::operator=( const TaskProvider & provider )
    {
        if( this != &provider )
            _threadPool = provider._threadPool;

        return (*this);
    }

    void TaskProvider::setThreadPool( ThreadPool * pool )
    {
        _threadPool = pool;
    }

    void TaskProvider::_run( size_t taskCount )
    {
        if( _ready() && taskCount > 0 ) {
            _taskCount          = taskCount;
            _givenTaskCount     = 0;
            _completedTaskCount = 0;

            _threadPool->add( this, _taskCount );
        }
    }

    bool TaskProvider::_ready() const
    {
        return AbstractTaskProvider::_ready() && _threadPool != nullptr;
    }


    ThreadPool::ThreadPool( size_t threads )
        : _runningThreadCount ( 0 )
        , _threadCount        ( 0 )
        , _threadsCreated     ( false )
    {
        resize( threads );
    }

    ThreadPool::~ThreadPool()
    {
        stop();
    }

    void ThreadPool::resize( size_t threads )
    {
        if( threads == 0 )
            throw imageException( "Try to set zero threads in thread pool" );

        if( threads > threadCount() ) {
            _taskInfo.lock();
            _run.resize( threads, 1 );
            _exit.resize( threads, 0 );
            _taskInfo.unlock();

            _threadCount = threads;

            while( threads > threadCount() )
                _worker.push_back( std::thread ( ThreadPool::_workerThread, this, threadCount() ) );

            std::unique_lock < std::mutex > _mutexLock( _creation );
            _completeCreation.wait( _mutexLock, [&] { return _threadsCreated; } );
        }
        else if( threads < threadCount() ) {
            _taskInfo.lock();

            std::fill( _exit.begin() + threads, _exit.end(), 1 );
            std::fill( _run.begin() + threads, _run.end(), 1 );
            _waiting.notify_all();

            _taskInfo.unlock();

            while( threads < threadCount() ) {
                _worker.back().join();

                _worker.pop_back();

                _taskInfo.lock();

                _exit.pop_back();
                _run.pop_back();

                _taskInfo.unlock();
            }
        }
    }

    size_t ThreadPool::threadCount() const
    {
        return _worker.size();
    }

    void ThreadPool::add( AbstractTaskProvider * provider, size_t taskCount )
    {
        if( taskCount == 0 )
            throw imageException( "Try to add zero tasks to thread pool" );

        if( threadCount() == 0 )
            throw imageException( "No threads in thread pool" );

        provider->_completion.lock();
        provider->_running = true;
        provider->_completion.unlock();

        _taskInfo.lock();

        _task.insert( _task.end(), taskCount, provider );

        std::fill( _run.begin(), _run.end(), 1 );
        _waiting.notify_all();

        _taskInfo.unlock();
    }

    bool ThreadPool::empty()
    {
        _taskInfo.lock();

        bool emp = _task.empty();

        _taskInfo.unlock();

        return emp;
    }

    void ThreadPool::remove( AbstractTaskProvider * provider )
    {
        _taskInfo.lock();

        _task.remove( provider );

        _taskInfo.unlock();
    }

    void ThreadPool::clear()
    {
        _taskInfo.lock();
        // complete all tasks without real computations. It helps to avoid a deadlock in a case when thread pool is destroyed
        std::for_each( _task.begin(), _task.end(), []( AbstractTaskProvider * task ) { task->_task( true ); } );
        _task.clear();

        _taskInfo.unlock();
    }

    void ThreadPool::stop()
    {
        clear();

        if( !_worker.empty() ) {
            _taskInfo.lock();

            std::fill( _exit.begin(), _exit.end(), 1 );
            std::fill( _run.begin(), _run.end(), 1 );
            _waiting.notify_all();

            _taskInfo.unlock();

            for( std::vector < std::thread >::iterator thread = _worker.begin(); thread != _worker.end(); ++thread )
                thread->join();

            _worker.clear();
        }
    }

    void ThreadPool::_workerThread( ThreadPool * pool, size_t threadId )
    {
        if( ++(pool->_runningThreadCount) == pool->_threadCount ) {
            pool->_creation.lock();
            pool->_threadsCreated = true;
            pool->_completeCreation.notify_one();
            pool->_creation.unlock();
        }

        while( !pool->_exit[threadId] ) {
            std::unique_lock < std::mutex > _mutexLock( pool->_taskInfo );
            pool->_waiting.wait( _mutexLock, [&] { return pool->_run[threadId]; } );
            _mutexLock.unlock();

            if( pool->_exit[threadId] )
                break;

            pool->_taskInfo.lock();

            if( !pool->_task.empty() ) {
                AbstractTaskProvider * task = pool->_task.front();

                pool->_task.pop_front();

                pool->_taskInfo.unlock();

                task->_taskRun( false );
            }
            else {
                pool->_run[threadId] = 0;

                pool->_taskInfo.unlock();
            }
        }

        --(pool->_runningThreadCount);
    }


    TaskProviderSingleton::TaskProviderSingleton()
    {
    }

    TaskProviderSingleton::TaskProviderSingleton( const TaskProviderSingleton & )
    {
    }

    TaskProviderSingleton::~TaskProviderSingleton()
    {
        ThreadPoolMonoid::instance().remove( this );
    }

    TaskProviderSingleton & TaskProviderSingleton::operator=( const TaskProviderSingleton & )
    {
        return (*this);
    }

    void TaskProviderSingleton::_run( size_t taskCount )
    {
        if( _ready() && taskCount > 0 ) {
            _taskCount          = taskCount;
            _givenTaskCount     = 0;
            _completedTaskCount = 0;

            ThreadPoolMonoid::instance().add( this, _taskCount );
        }
    }
};
