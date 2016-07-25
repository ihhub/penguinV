#pragma once

#include <exception>
#include <string>

class imageException : public std::exception
{
public:

	imageException()
	{
		_name = "unknown image library exception";
	}

	imageException(const char * message)
	{
		_name = message;
	}

	imageException(const imageException & ex)
		: std::exception( ex )
	{
		_name = ex._name;
	}

	~imageException()
	{

	}

	imageException & operator=(const imageException & ex)
	{
		std::exception::operator=( ex );

		_name = ex._name;

		return (*this);
	}

	const char * what()
	{
		return _name.c_str();
	}

private:
	std::string _name;
};
