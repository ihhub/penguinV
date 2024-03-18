/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2024                                             *
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

#if defined( __APPLE__ ) || defined( __MACOSX )
#include <OpenCL/cl.h>
#else

#define CL_TARGET_OPENCL_VERSION 210

#include <CL/cl.h>
#endif

#include "opencl_memory.h"
#include <list>
#include <string>
#include <vector>

namespace multiCL
{
    class OpenCLDeviceManager;

    // This is a shortcut (facade) namespace to access memory allocator for devices
    namespace MemoryManager
    {
        // Returns memory allocator for current thread
        // By default it will return an allocator for device with ID = 0 otherwise an ID what is actual for current thread
        MemoryAllocator & memory();

        // Returns memory allocator for specified device ID
        MemoryAllocator & memory( const uint32_t deviceId );

        void memorySet( cl_mem data, const void * pattern, size_t patternSize, size_t offset, size_t size );
    }

    class OpenCLContext
    {
    public:
        explicit OpenCLContext( cl_device_id deviceId );

        explicit OpenCLContext( cl_context context )
            : _context( context )
        {
            // Do nothing.
        }

        OpenCLContext( const OpenCLContext & ) = delete;

        OpenCLContext & operator=( const OpenCLContext & ) = delete;

        ~OpenCLContext();

        cl_context operator()() const
        {
            return _context;
        }

    private:
        cl_context _context;
    };

    class OpenCLProgram
    {
    public:
        OpenCLProgram( const OpenCLContext & context, const char * program );

        explicit OpenCLProgram( cl_program program )
            : _program( program )
        {
            // Do nothing.
        }

        OpenCLProgram( OpenCLProgram && program )
            : _program( NULL )
        {
            std::swap( _program, program._program );
        }

        OpenCLProgram( const OpenCLProgram & ) = delete;

        OpenCLProgram & operator=( const OpenCLProgram & ) = delete;

        ~OpenCLProgram();

        cl_program operator()() const
        {
            return _program;
        }

    private:
        cl_program _program;
    };

    class OpenCLKernel
    {
    public:
        OpenCLKernel( const OpenCLProgram & program, const std::string & name );

        explicit OpenCLKernel( cl_kernel kernel )
            : _kernel( kernel )
            , _parameterId( 0 )
        {
            // Do nothing.
        }

        OpenCLKernel( const OpenCLKernel & ) = delete;

        OpenCLKernel & operator=( const OpenCLKernel & ) = delete;

        ~OpenCLKernel();

        cl_kernel operator()() const
        {
            return _kernel;
        }

        template <typename T>
        void setArgument( T value )
        {
            _setArgument( sizeof( T ), &value );
        }

        template <typename T, typename... Args>
        void setArgument( T value, Args... args )
        {
            setArgument( value );
            setArgument( args... );
        }

        void reset()
        {
            _parameterId = 0;
        }

    private:
        cl_kernel _kernel;
        cl_uint _parameterId;

        void _setArgument( size_t size, const void * data );
    };

    class OpenCLQueue
    {
    public:
        OpenCLQueue( const OpenCLContext & context, cl_device_id deviceId );

        explicit OpenCLQueue( cl_command_queue queue )
            : _commandQueue( queue )
        {
            // Do nothing.
        }

        OpenCLQueue( const OpenCLQueue & ) = delete;

        OpenCLQueue & operator=( const OpenCLQueue & ) = delete;

        ~OpenCLQueue();

        cl_command_queue operator()() const
        {
            return _commandQueue;
        }

        void synchronize();

    private:
        cl_command_queue _commandQueue;
    };

    class OpenCLDevice
    {
    public:
        friend class OpenCLDeviceManager;

        OpenCLDevice( const OpenCLDevice & ) = delete;

        OpenCLDevice & operator=( const OpenCLDevice & ) = delete;

        ~OpenCLDevice();

        // Device information
        cl_device_id deviceId() const
        {
            return _deviceId;
        }

        // Maximum available number of threads per block.
        size_t threadsPerBlock( const OpenCLKernel & kernel ) const;

        // Total available memory in bytes.
        uint64_t totalMemorySize() const;

        std::string name() const;

        std::string computeCapability() const;

        // Device manipulation
        void synchronize(); // synchronize all operations on device with CPU

        OpenCLContext & context()
        {
            return _context;
        }

        const OpenCLContext & context() const
        {
            return _context;
        }

        // Current queue ID which is used as a default value in queue() function.
        size_t currentQueueId() const
        {
            return _currentQueueId;
        }

        void setCurrentQueueId( const size_t queueId )
        {
            if ( _currentQueueId != queueId && queueId < _queue.size() ) {
                _currentQueueId = queueId;
            }
        }

        // A reference to current queue.
        OpenCLQueue & queue()
        {
            return *( _queue[_currentQueueId] );
        }

        const OpenCLQueue & queue() const
        {
            return *( _queue[_currentQueueId] );
        }

        // A reference to queue with specified ID.
        OpenCLQueue & queue( size_t queueId )
        {
            return *( _queue[queueId] );
        }

        const OpenCLQueue & queue( size_t queueId ) const
        {
            return *( _queue[queueId] );
        }

        // Total number of queues.
        size_t queueCount() const
        {
            return _queue.size();
        }

        void setQueueCount( size_t queueCount );

        // Memory allocator associated with device.
        MemoryAllocator & allocator()
        {
            return *_allocator;
        }

        const MemoryAllocator & allocator() const
        {
            return *_allocator;
        }

    private:
        cl_device_id _deviceId;
        OpenCLContext _context;

        size_t _currentQueueId;

        // Array of queues within the device.
        std::vector<OpenCLQueue *> _queue;

        // Memory allocator on the current device.
        MemoryAllocator * _allocator;

        explicit OpenCLDevice( cl_device_id deviceId );
    };

    class OpenCLDeviceManager
    {
    public:
        static OpenCLDeviceManager & instance();

        // Initializes all devices available in system.
        void initializeDevices();

        // Initializes a device with specified ID.
        void initializeDevice( uint32_t deviceId );

        // Closes initialized device with specified ID.
        void closeDevice( uint32_t deviceId );

        // Closes all devices initialized by manager.
        void closeDevices();

        // Initialized devices via manager.
        uint32_t deviceCount() const
        {
            return static_cast<uint32_t>( _device.size() );
        }

        // Maximum available devices in the system.
        uint32_t supportedDeviceCount() const
        {
            return static_cast<uint32_t>( _supportedDeviceId.size() );
        }

        // Returns device within current thread.
        OpenCLDevice & device();

        // Returns device within current thread.
        const OpenCLDevice & device() const;

        // Returns device with specified ID.
        OpenCLDevice & device( uint32_t deviceId );

        // Returns device with specified ID.
        const OpenCLDevice & device( uint32_t deviceId ) const;

        // Set device with specified ID as a active device in current thread.
        void setActiveDevice( uint32_t deviceId );

        void resetSupportedDevice();

    private:
        OpenCLDeviceManager()
        {
            resetSupportedDevice();
        }

        ~OpenCLDeviceManager()
        {
            closeDevices();
        }

        std::vector<cl_device_id> _supportedDeviceId;

        // A list of initialized devices.
        std::list<OpenCLDevice *> _device;
    };
}
