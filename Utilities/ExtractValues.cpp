/*
 * ExtractValues.cxx
 *
 *  Created on: Jul 6, 2015
 *      Author: vgupta
 */

#include "iostream"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTransform.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "../inc/ComposeImage.h"
#include "../inc/MapFilterLR2HR.h"
#include "../GetPot/GetPot"
#include "itkImageRegionIterator.h"
#include "fstream"

//#include "../src/itkMapLR2HR.h"
using namespace std;
int main(int argc, char* *argv)
{


	     GetPot   cl(argc, argv);
	     if( cl.size() == 1 || cl.search(2, "--help", "-h") )
	      {
	          std::cout << "Not enough arguments" << std::endl;
	          return -1;
	      }

	     const string image_N = cl.follow("NoFile", 1 , "-i");
	     const string maskImage_N = cl.follow("NoFile", 1 , "-m");
	     const string out_n = cl.follow("NoFile",1, "-o");

	     typedef itk::Image<float, 3> ImageType;
	     typedef itk::ImageFileReader<ImageType> ReaderType;
	     typedef itk::ImageFileWriter<ImageType> WriterType;

	     ReaderType::Pointer reader = ReaderType::New();
	     reader->SetFileName(image_N);
	     reader->Update();
	     ImageType::Pointer image = reader->GetOutput();

	     ReaderType::Pointer mask_reader = ReaderType::New();
	     mask_reader->SetFileName(maskImage_N);
	     mask_reader->Update();

	     ImageType::Pointer mask = mask_reader ->GetOutput();

	     typedef itk::ImageRegionIterator<ImageType> IteratorType;
	     IteratorType itFA(image, image->GetLargestPossibleRegion());
	     IteratorType itMask(mask, mask->GetLargestPossibleRegion());

	     ofstream file;
	     file.open(out_n);


	     int count=1;



	     for (itMask.GoToBegin(), itFA.GoToBegin(); !itMask.IsAtEnd(), !itFA.IsAtEnd(); ++itMask, ++itFA)
	     {
	    	 if (itMask.Get() >=0.85)
	    	 {
	    		 ImageType::PointType point;
	    		 ImageType::IndexType index;
	    		 index = itMask.GetIndex();
//	    	   	 image->TransformIndexToPhysicalPoint(index, point);

	    	   	 file << itFA.Get() << std::endl;
	    	   	 count++;
	    	 }

	     }

	     file.close();


	     return 0;
}

