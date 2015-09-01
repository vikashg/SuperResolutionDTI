/*
 * TestComposeImage.cxx
 *
 *  Created on: Jul 4, 2015
 *      Author: vgupta
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "../src/itkComposeImage.h"

using namespace std;
int main(int argc, char* *argv)
{


	     GetPot   cl(argc, argv);
	     if( cl.size() == 1 || cl.search(2, "--help", "-h") )
	      {
	          std::cout << "Not enough arguments" << std::endl;
	          return -1;
	      }

	    const string LR_file_n = cl.follow("NoFile", 1 , "-LR");
	 	const string HR_file_n = cl.follow("NoFile", 1 , "-HR");
	 	const string output_file_n = cl.follow("Nofile.nii.gz", 1, "-o");
	 	const string trans_n   = cl.follow("NoFile", 1, "-t");



}



