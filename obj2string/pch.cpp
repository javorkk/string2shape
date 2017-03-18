#include "pch.h"

//////////////////////////////////////////////////////////////////////////
//String utilities
//////////////////////////////////////////////////////////////////////////
std::string itoa(const int a)
{
	std::stringstream ss;
	ss << a;
	return ss.str();
}

std::string ftoa(const float a)
{
	std::stringstream ss;
	ss.precision(2);
	ss.setf(std::ios::fixed, std::ios::floatfield);
	ss << a;
	return ss.str();
}

std::string cutComments(const std::string & aLine, const char * aToken)
{
	std::string token(aToken);
	const size_t commentStart = aLine.find(token, 0);

	if (commentStart == std::string::npos)
	{
		//line does not contain comment token
		const size_t lineBegin = aLine.find_first_not_of(" \t", 0);
		if (lineBegin < aLine.size())
		{
			return aLine.substr(lineBegin, aLine.size());
		}
		else
		{
			//line contains only whitespace
			return std::string("");
		}
	}
	else
	{
		//line contains comment token
		const size_t lineBegin = aLine.find_first_not_of(" \t", 0);
		return aLine.substr(lineBegin, commentStart - lineBegin);
	}
}
