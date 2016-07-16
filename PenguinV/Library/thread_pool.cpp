#include <algorithm>
#include "thread_pool.h"
#include "image_exception.h"

TaskProvider::TaskProvider()
	: _taskCount         (0)
	, _givenTaskCount    (0)
	, _completedTaskCount(0)
	, _running           (false)
	, _threadPool        (nullptr)
{
}

TaskProvider::TaskProvider(ThreadPool * pool)
	: _taskCount         (0)
	, _givenTaskCount    (0)
	, _completedTaskCount(0)
	, _running           (false)
	, _threadPool        (pool)
{
}

TaskProvider::TaskProvider(const TaskProvider & provider)
	: _taskCount         (0)
	, _givenTaskCount    (0)
	, _completedTaskCount(0)
	, _running           (false)
	, _threadPool        (provider._threadPool)
{
}

TaskProvider::~TaskProvider()
{
	if( _threadPool != nullptr )
		_threadPool->remove( this );

	_wait();
}

TaskProvider & TaskProvider::operator=(const TaskProvider & provider)
{
	if( this != &provider ) {
		_threadPool = provider._threadPool;
	}

	return (*this);
}

void TaskProvider::setThreadPool( ThreadPool * pool )
{
	_threadPool = pool;
}

void TaskProvider::_taskRun(bool skip)
{
	size_t taskId = _givenTaskCount++;

	if( taskId < _taskCount ) {
		if( !skip )
			_task(taskId);

		size_t completedTasks = (++_completedTaskCount);

		if( completedTasks == _taskCount ) {
			_running = false;
			_waiting.notify_one();
		}
	}
}

void TaskProvider::_run( size_t taskCount )
{
	if( _ready() && taskCount > 0 ) {
		_taskCount          = taskCount;
		_givenTaskCount     = 0;
		_completedTaskCount = 0;
		_running            = true;

		_threadPool->add( this, _taskCount );
	}
}

void TaskProvider::_wait()
{
	std::unique_lock < std::mutex > _mutexLock( _completion );
	_waiting.wait( _mutexLock, [&] { return !_running; } );
}

bool TaskProvider::_ready() const
{
	return !_running && _threadPool != nullptr;
}


ThreadPool::ThreadPool(size_t threads)
	: _threadId      (0)
	, _threadCount   (0)
	, _threadsCreated(false)
{
	resize( threads );
}

ThreadPool::~ThreadPool()
{
	_stop();
}

void ThreadPool::resize( size_t threads )
{
	if( threads == 0 )
		throw imageException("Try to set zero threads in thread pool");

	if( threads > threadCount() ) {
		_taskInfo.lock();
		_run.resize( threads, 1 );
		_exit.resize( threads, 0 );
		_taskInfo.unlock();

		_threadCount = threads;

		while( threads > threadCount() )
			_worker.push_back( std::thread (ThreadPool::_workerThread, this ) );

		std::unique_lock < std::mutex > _mutexLock( _creation );
		_completeCreation.wait( _mutexLock, [&] { return _threadsCreated; } );

	}
	else if( threads < threadCount() ) {
		_taskInfo.lock();
		std::fill( _exit.begin() + threads, _exit.end(), 1 );
		std::fill(  _run.begin() + threads,  _run.end(), 1 );
		_taskInfo.unlock();

		_waiting.notify_all();

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

void ThreadPool::add( TaskProvider * provider, size_t taskCount )
{
	if( taskCount == 0 )
		throw imageException("Try to add zero tasks to thread pool");

	_taskInfo.lock();

	_task.insert( _task.end(), taskCount, provider );

	std::fill( _run.begin(), _run.end(), 1 );

	_taskInfo.unlock();

	_waiting.notify_all();
}

bool ThreadPool::empty()
{
	_taskInfo.lock();

	bool emp = _task.empty();

	_taskInfo.unlock();

	return emp;
}

void ThreadPool::remove( TaskProvider * provider )
{
	_taskInfo.lock();

	_task.remove( provider );

	_taskInfo.unlock();
}

void ThreadPool::clear()
{
	_taskInfo.lock();

	std::for_each( _task.begin(), _task.end(), [](TaskProvider * task) { task->_task( true ); } );
	_task.clear();

	_taskInfo.unlock();
}

void ThreadPool::_stop()
{
	clear();

	if( !_worker.empty() ) {
		_taskInfo.lock();
		std::fill( _exit.begin(), _exit.end(), 1 );
		std::fill(  _run.begin(),  _run.end(), 1 );
		_taskInfo.unlock();

		_waiting.notify_all();

		for( std::vector < std::thread >::iterator thread = _worker.begin(); thread != _worker.end(); ++thread )
			thread->join();
	}

}

void ThreadPool::_workerThread( ThreadPool * pool )
{
	size_t threadId = pool->_threadId++;

	if( (threadId + 1) == pool->_threadCount ) {
		pool->_threadsCreated = true;
		pool->_completeCreation.notify_one();
	}

	while( !pool->_exit[threadId] ) {

		std::unique_lock < std::mutex > _mutexLock( pool->_runTask );
		pool->_waiting.wait( _mutexLock, [&] { return pool->_run[threadId]; } );
		_mutexLock.unlock();

		if( pool->_exit[threadId] )
			break;

		pool->_taskInfo.lock();

		if( !pool->_task.empty() ) {
			TaskProvider * task = pool->_task.front();

			pool->_task.pop_front();

			pool->_taskInfo.unlock();

			task->_taskRun( false );
		}
		else {
			pool->_run[threadId] = 0;

			pool->_taskInfo.unlock();
		}
		
	}

	--(pool->_threadId);
}
