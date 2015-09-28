/*
 * ImageDispplayProperties.cxx
 *
 *  Created on: Jul 7, 2015
 *      Author: vgupta
 */




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

	     typedef itk::Image<float, 3> ImageType;
	     typedef itk::ImageFileReader<ImageType> ReaderType;

	     ReaderType::Pointer reader = ReaderType::New();
	     reader->SetFileName(image_N);
	     reader->Update();
	     ImageType::Pointer image = reader->GetOutput();

	     std::cout << "Origin " << image->GetOrigin() << std::endl;
	     std::cout << "Spacing " << image->GetSpacing() << std::endl;
	     std::cout << "Size "   << image->GetLargestPossibleRegion().GetSize() << std::endl;
	     std::cout << "Directions " << image->GetDirection() << std::endl;

	    ImageType::IndexType tempIndex, tempIndex1;
	    tempIndex.Fill(0); tempIndex1.Fill(1);

		
	


	     return 0;
}

