#ifdef _MSC_VER
#pragma once
#endif

#ifndef GRAPH2STRING_H_18BBEE2E_4283_4957_8B2C_8AC609108C63
#define GRAPH2STRING_H_18BBEE2E_4283_4957_8B2C_8AC609108C63

#include <cuda_runtime_api.h>
#include <vector>
#include <set>
#include <string>

#include "WFObject.h"
#include "Graph.h"

class GraphToStringConverter
{
	std::vector<std::string> mAlphabet;
	std::string mBranchStart;
	std::string mBranchEnd;
	std::string mNumberSeparator;
public:
	__host__  GraphToStringConverter()
	{
		mBranchStart = std::string("(");
		mBranchEnd = std::string(")");
		mNumberSeparator = std::string("%");
		mAlphabet.push_back("|zero|");
		mAlphabet.push_back("A");
		mAlphabet.push_back("B");
		mAlphabet.push_back("C");
		mAlphabet.push_back("D");
		mAlphabet.push_back("E");
		mAlphabet.push_back("F");
		mAlphabet.push_back("G");
		mAlphabet.push_back("H");
		mAlphabet.push_back("I");
		mAlphabet.push_back("J");
		mAlphabet.push_back("K");
		mAlphabet.push_back("L");
		mAlphabet.push_back("M");
		mAlphabet.push_back("N");
		mAlphabet.push_back("O");
		mAlphabet.push_back("P");
		mAlphabet.push_back("Q");
		mAlphabet.push_back("R");
		mAlphabet.push_back("S");
		mAlphabet.push_back("T");
		mAlphabet.push_back("U");
		mAlphabet.push_back("V");
		mAlphabet.push_back("W");
		mAlphabet.push_back("X");
		mAlphabet.push_back("Y");
		mAlphabet.push_back("Z");
		mAlphabet.push_back("a");
		mAlphabet.push_back("b");
		mAlphabet.push_back("c");
		mAlphabet.push_back("d");
		mAlphabet.push_back("e");
		mAlphabet.push_back("f");
		mAlphabet.push_back("g");
		mAlphabet.push_back("h");
		mAlphabet.push_back("i");
		mAlphabet.push_back("j");
		mAlphabet.push_back("k");
		mAlphabet.push_back("l");
		mAlphabet.push_back("m");
		mAlphabet.push_back("n");
		mAlphabet.push_back("o");
		mAlphabet.push_back("p");
		mAlphabet.push_back("q");
		mAlphabet.push_back("r");
		mAlphabet.push_back("s");
		mAlphabet.push_back("t");
		mAlphabet.push_back("u");
		mAlphabet.push_back("v");
		mAlphabet.push_back("w");
		mAlphabet.push_back("x");
		mAlphabet.push_back("y");
		mAlphabet.push_back("z");

	}
	
	__host__ std::string depthFirstTraverse(
		unsigned int v,
		thrust::host_vector<unsigned int>& visited,
		unsigned int parent,
		thrust::host_vector<unsigned int>& intervalsHost,
		thrust::host_vector<unsigned int>& adjacencyValsHost,
		thrust::host_vector<Graph::EdgeType>& adjacencyMatrixType,
		thrust::host_vector<unsigned int>& cycleIds,
		thrust::host_vector<unsigned int>& nodeTypeIds);
	__host__ std::string toString(Graph& aGraph, thrust::host_vector<unsigned int>& aNodeTypes);
	__host__ std::string operator()(WFObject& aObj, Graph& aGraph);
};

class GrammarCheck
{
	unsigned int mNumTypes;
	std::vector< std::vector<unsigned int> > mNeighborCounts;
	std::set< std::pair<unsigned int, unsigned int> > mNeighborTypes;
	std::set< std::pair<unsigned int, std::vector<unsigned int> > > mNeighborTypeCounts;
public:
	__host__ GrammarCheck():mNumTypes(0u)
	{}

	__host__ void init(
		thrust::host_vector<unsigned int>& aIntervals,
		thrust::host_vector<unsigned int>& aNbrIds,
	    thrust::host_vector<unsigned int>& aNodeTypes);
	__host__ void init(
		WFObject& aObj,
		thrust::device_vector<unsigned int>& aIntervals,
		thrust::device_vector<unsigned int>& aNbrIds);
	__host__ bool check(
		thrust::host_vector<unsigned int>& aIntervals,
		thrust::host_vector<unsigned int>& aNbrIds,
		thrust::host_vector<unsigned int>& aNodeTypes);

	__host__ void cleanup();

};
#endif // GRAPH2STRING_H_18BBEE2E_4283_4957_8B2C_8AC609108C63
