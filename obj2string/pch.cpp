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

std::string getDirName(const std::string & _name)
{
	std::string objDir;
#if _MSC_VER >= 1400
	char fileDir[4096];
	_splitpath_s(_name.c_str(), NULL, 0, fileDir, sizeof(fileDir), NULL, 0, NULL, 0);
	objDir = fileDir;
#endif

#ifndef _WIN32
	char *fnCopy = strdup(_name.c_str());
	const char* dirName = dirname(fnCopy);
	objDir = dirName;
	objDir.append("/");
	free(fnCopy);
	//std::cerr << "Dirname: " << objDir << "\n";
#endif // _WIN32

	return objDir;
}
