#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool;

class TaskProvider
{
public:
	friend class ThreadPool;

	TaskProvider();
	TaskProvider(ThreadPool * pool);
	TaskProvider(const TaskProvider & provider);
	virtual ~TaskProvider();

	TaskProvider & operator=(const TaskProvider & provider);

	void setThreadPool( ThreadPool * pool );

protected:
	virtual void _task(size_t) = 0; // this function must be overrided in child class and should contain code specific to task ID
									  // parameter in function is task ID. This function is called by thread pool
	
	void _run( size_t taskCount );
	void _wait();

	bool _ready() const;

private:
	std::atomic < size_t > _taskCount;          // number of tasks to do
	std::atomic < size_t > _givenTaskCount;     // number of tasks what were given to thread pool
	std::atomic < size_t > _completedTaskCount; // number of completed tasks

	bool _running;                    // boolean variable specifies the state of tasks processing
	std::mutex _completion;           // mutex for synchronization reporting about completion of all tasks
							          // this mutex is waited in _wait() function
	std::condition_variable _waiting; // condition variable for verification that all tasks are really completed

	ThreadPool * _threadPool; // a pointer to thread pool

	void _taskRun(bool skip); // function called by thread pool
};

class ThreadPool
{
public:
	ThreadPool() { };
	ThreadPool(size_t threads);
	~ThreadPool();

	void resize( size_t threads );
	size_t threadCount() const;

	void add( TaskProvider * provider, size_t taskCount );
	void remove( TaskProvider * provider );
	bool empty();
	void clear();

private:
	ThreadPool & operator=(const ThreadPool &);
	ThreadPool(const ThreadPool &);

	std::vector < std::thread > _worker; // an array of worker threads
	std::mutex _runTask;                 // mutex to synchronize threads
	std::vector < uint8_t > _run;        // indicator for threads to run tasks
	std::vector < uint8_t > _exit;       // indicator for threads to close themselfs
	std::condition_variable _waiting;    // condition variable for synchronization of threads

	std::mutex _creation;                       // mutex for thread creation verification
	std::atomic < size_t > _runningThreadCount; // variable used to calculate a number of running threads
	std::condition_variable _completeCreation;  // condition variable for verification that all threads are created
	std::size_t _threadCount;                   // current number of threads in pool
	bool _threadsCreated;                       // indicator for pool that all threads are created

	std::list < TaskProvider * > _task; // a list of tasks to perform
	std::mutex _taskInfo;               // mutex for synchronization between threads and pool to manage tasks

	void _stop();

	static void _workerThread( ThreadPool * pool, size_t threadId );
};
