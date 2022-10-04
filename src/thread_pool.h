/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool;

// General abstract class to work with thread pool
class AbstractTaskProvider
{
public:
    friend class ThreadPool;
    friend class TaskProvider;
    friend class TaskProviderSingleton;

    AbstractTaskProvider();
    AbstractTaskProvider( const AbstractTaskProvider & );
    virtual ~AbstractTaskProvider();

    AbstractTaskProvider & operator=( const AbstractTaskProvider & );

protected:
    virtual void _task( size_t ) = 0; // this function must be overrided in child class and should contain a code specific to task ID
                                      // parameter in the function is task ID. This function must be called by thread pool
    bool _wait(); // waits for all task execution completions. Returns true in case of success, false when an exception is raised

    bool _ready() const; // this function tells whether class is able to use thread pool
private:
    std::atomic<size_t> _taskCount; // number of tasks to do
    std::atomic<size_t> _givenTaskCount; // number of tasks which were given to thread pool
    std::atomic<size_t> _completedTaskCount; // number of completed tasks

    bool _running; // boolean variable specifies the state of tasks processing
    std::mutex _completion; // mutex for synchronization reporting about completion of all tasks
                            // this mutex is waited in _wait() function
    std::condition_variable _waiting; // condition variable for verification that all tasks are really completed

    bool _exceptionRaised; // notifies whether an exception raised during task execution

    void _taskRun( bool skip ); // function is called only by thread pool to call _task() function and increment counters
};

// Concrete class of task provider for case when thread pool is not a singleton
class TaskProvider : public AbstractTaskProvider
{
public:
    TaskProvider();
    explicit TaskProvider( ThreadPool * pool );
    TaskProvider( const TaskProvider & provider );
    virtual ~TaskProvider();

    TaskProvider & operator=( const TaskProvider & provider );

    void setThreadPool( ThreadPool * pool );

protected:
    void _run( size_t taskCount );

    bool _ready() const;

private:
    ThreadPool * _threadPool; // a pointer to a thread pool
};

class ThreadPool
{
public:
    explicit ThreadPool( size_t threads = 0u );
    ThreadPool & operator=( const ThreadPool & ) = delete;
    ThreadPool( const ThreadPool & ) = delete;
    ~ThreadPool();

    void resize( size_t threads );
    size_t threadCount() const;

    void add( AbstractTaskProvider * provider, size_t taskCount ); // add tasks for specific provider
    void remove( AbstractTaskProvider * provider ); // remove all tasks related to specific provider
    bool empty(); // tells whether thread pool contains any tasks
    void clear(); // remove all tasks from thread pool

    void stop(); // stop all working threads
private:
    std::vector<std::thread> _worker; // an array of worker threads
    std::vector<uint8_t> _run; // indicator for threads to run tasks
    std::vector<uint8_t> _exit; // indicator for threads to close themselfs
    std::condition_variable _waiting; // condition variable for synchronization of threads

    std::mutex _creation; // mutex for thread creation verification
    std::atomic<size_t> _runningThreadCount; // variable used to calculate a number of running threads
    std::condition_variable _completeCreation; // condition variable for verification that all threads are created
    std::size_t _threadCount; // current number of threads in pool
    bool _threadsCreated; // indicator for pool that all threads are created

    std::list<AbstractTaskProvider *> _task; // a list of tasks to perform
    std::mutex _taskInfo; // mutex for synchronization between threads and pool to manage tasks

    static void _workerThread( ThreadPool * pool, size_t threadId );
};

// Thread pool singleton (or monoid class) for whole application
// In most situations thread pool must be one
class ThreadPoolMonoid
{
public:
    static ThreadPool & instance(); // function returns a reference to global (static) thread pool

    ThreadPoolMonoid & operator=( const ThreadPoolMonoid & ) = delete;
    ThreadPoolMonoid( const ThreadPoolMonoid & ) = delete;
    ~ThreadPoolMonoid() {}

private:
    ThreadPoolMonoid() {}

    ThreadPool _pool; // one and only thread pool
};

// Concrete class of task provider with thread pool singleton
class TaskProviderSingleton : public AbstractTaskProvider
{
public:
    TaskProviderSingleton();
    TaskProviderSingleton( const TaskProviderSingleton & provider );
    virtual ~TaskProviderSingleton();

    TaskProviderSingleton & operator=( const TaskProviderSingleton & );

protected:
    void _run( size_t taskCount );
};
