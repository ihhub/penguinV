// Example application of library's thread pool utilization
#include <iostream>
#include "../../Library/image_exception.h"
#include "../../Library/image_function.h"
#include "../../Library/thread_pool.h"

void example1();
void example2();

int main()
{
	try // <---- do not forget to put your code into try.. catch block!
	{
		std::cout << "----------" << std::endl
			<< "Thread pool as a class object. The pool will be destroyed at the end of function" << std::endl
			<< "----------" << std::endl;
		example1();
		std::cout << "----------" << std::endl
			<< "Thread pool as singleton. The pool will be destroyed at the end of application." << std::endl
			<< "----------" << std::endl;
		example2();
	} catch(imageException & ex) {
		// uh-oh, something went wrong!
		std::cout << "Exception " << ex.what() << " raised. Do your black magic to recover..." << std::endl;
		// your magic code must be here to recover from bad things
		return 0;
	} catch(std::exception & ex) {
		// uh-oh, something terrible happen!
		// it might be that you compiled code in linux without threading parameters
		std::cout << "Something terrible happen (" << ex.what() << "). Do your black magic to recover..." << std::endl;
		// your magic code must be here to recover from terrible things
		return 0;
	} catch(...) {
		// uh-oh, something really terrible happen!
		std::cout << "Something really terrible happen. No idea what it is. Do your black magic to recover..." << std::endl;
		// your magic code must be here to recover from terrible things
		return 0;
	}

	std::cout << "Everything went fine." << std::endl;

	return 0;
}

// Abstract class for example 1 and 2
class AbstractTaskGiver
{
public:
	AbstractTaskGiver()
	{
		// we use static variables to set correct ID
		static uint32_t objectCounter = 0;
		_objectId = (++objectCounter);
	};

protected:
	void completeTask(size_t taskId)
	{
		for( int count = 0; count < _repeatCount[taskId]; ++count ) {
			// This is just some random code for image processing to make your CPU busy
			Bitmap_Image::Image image1( 1024, 1024 );

			image1.fill( 255 );

			Bitmap_Image::Image image2 = Image_Function::Invert( image1 );

			Bitmap_Image::Image image3 = Image_Function::Maximum( image1, image2 );
		}

		_message.lock();
		std::cout << "Object " << _objectId << ": task #" << taskId << " completed." << std::endl;
		_message.unlock();
	};

	size_t createTasks()
	{
		// create random number of tasks
		_repeatCount.resize( (rand() % 32) + 1 );

		for( std::vector < int >::iterator count = _repeatCount.begin(); count != _repeatCount.end(); ++count )
			*count = rand() % 256 + 64;

		_message.lock();
		std::cout << "Object " << _objectId << ": " << _repeatCount.size() << " tasks are created" << std::endl;
		_message.unlock();

		return _repeatCount.size();
	};

private:
	std::vector < int > _repeatCount; // our spefic work to execute
	std::mutex _message;              // mutex just for synchronization of messages output
	uint32_t _objectId;               // object ID needed to show for what object task is completed
};

// Task giver class for example 1 inherited from TaskProvider
class TaskGiver1 : public Thread_Pool::TaskProvider, public AbstractTaskGiver
{
public:
	TaskGiver1(Thread_Pool::ThreadPool * pool)
		: TaskProvider(pool)
	{
	};
	
	virtual ~TaskGiver1() { };

	void run()
	{
		if(! _ready() ) {
			std::cout << "Task giver is not ready" << std::endl;
			return;
		}

		_run( createTasks() );
	};

	void wait()
	{
		_wait();
	};

protected:
	virtual void _task(size_t taskId)
	{
		completeTask( taskId );
	};
};

// This is task giver class for example 2 inherited from TaskProviderSingleton
// TaskGiver1 and TaskGiver2 are totally same for task execution point of view
class TaskGiver2 : public Thread_Pool::TaskProviderSingleton, public AbstractTaskGiver
{
public:	
	virtual ~TaskGiver2() { };

	void run()
	{
		if(! _ready() ) {
			std::cout << "Task giver is not ready" << std::endl;
			return;
		}

		_run( createTasks() );
	};

	void wait()
	{
		_wait();
	};

protected:
	virtual void _task(size_t taskId)
	{
		completeTask( taskId );
	};
};

void example1()
{
	// Create a thread pool with 4 worker threads
	std::cout << "Create thread pool with 4 threads" << std::endl;
	Thread_Pool::ThreadPool pool( 4 );

	// Create a task giver
	TaskGiver1 firstGiver( &pool );

	// Generate tasks and put it inside thread pool
	firstGiver.run();

	// Wait until all tasks are completed
	firstGiver.wait();

	// We decided to reduce number of threads held by thread pool to 2 due to some lack of resources
	std::cout << "Resize thread pool to 2 threads" << std::endl;
	pool.resize( 2 );

	// Generate another bunch of tasks and give them to thread pool
	firstGiver.run();

	// Wait until all tasks are completed
	firstGiver.wait();

	// Let's create another task giver object
	TaskGiver1 secondGiver( &pool );

	// Second giver generates tasks and feed them to thread pool
	secondGiver.run();

	// Suddenly our application got more power and we change number of thread in thread pool to 8
	// Starting from this line result displaying is unpredictable due to multithreading effects!
	std::cout << "Resize thread pool to 8 threads" << std::endl;
	pool.resize( 8 );

	// First giver generates tasks and feed them to thread pool
	firstGiver.run();

	// Wait until all tasks for first giver are completed
	firstGiver.wait();

	// Wait until all tasks for second giver are completed
	secondGiver.wait();
}

void example2()
{
	// Resize thread pool to 4 worker threads
	std::cout << "Resize thread pool to 4 threads" << std::endl;
	Thread_Pool::ThreadPoolMonoid::instance().resize( 4 );

	// Create a task giver
	TaskGiver2 firstGiver;

	// Generate tasks and put it inside thread pool
	firstGiver.run();

	// Wait until all tasks are completed
	firstGiver.wait();

	// We decided to reduce number of threads held by thread pool to 2 due to some lack of resources
	std::cout << "Resize thread pool to 2 threads" << std::endl;
	Thread_Pool::ThreadPoolMonoid::instance().resize( 2 );

	// Generate another bunch of tasks and give them to thread pool
	firstGiver.run();

	// Wait until all tasks are completed
	firstGiver.wait();

	// Let's create another task giver object
	TaskGiver2 secondGiver;

	// Second giver generates tasks and feed them to thread pool
	secondGiver.run();

	// Suddenly our application got more power and we change number of thread in thread pool to 8
	// Starting from this line result displaying is unpredictable due to multithreading effects!
	std::cout << "Resize thread pool to 8 threads" << std::endl;
	Thread_Pool::ThreadPoolMonoid::instance().resize( 8 );

	// First giver generates tasks and feed them to thread pool
	firstGiver.run();

	// Wait until all tasks for first giver are completed
	firstGiver.wait();

	// Wait until all tasks for second giver are completed
	secondGiver.wait();

	// We use singleton and we want to be sure that destructor will not have any exceptions
	// so we call stop() function what suppose to be called in destructor to clean all stuff. Profit!
	// Be aware that after this code no more threads will be in thread pool
	Thread_Pool::ThreadPoolMonoid::instance().stop();
}
