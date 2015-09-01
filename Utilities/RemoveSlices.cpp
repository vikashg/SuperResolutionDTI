/*
 * RemoveSlices.cpp
 *
 *  Created on: Jul 14, 2015
 *      Author: vgupta
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "../GetPot/GetPot"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"

using namespace std;
int main(int argc, char* *argv)
{


	     GetPot   cl(argc, argv);
	     if( cl.size() == 1 || cl.search(2, "--help", "-h") )
	      {
	          std::cout << "Not enough arguments" << std::endl;
	          return -1;
	      }

	     const string B0_image_n = cl.follow("NoFile", 1 , "-i");
	     const string output_Image_n = cl.follow("NoFile.nii.gz", 1, "-o");

	     typedef itk::Image<float, 3> ImageType;
	     typedef itk::ImageFileReader<ImageType> ReaderType;

	     typedef itk::ImageSliceIteratorWithIndex<ImageType>  SliceIteratorType;


	     ReaderType::Pointer reader  = ReaderType::New();
	     reader->SetFileName(B0_image_n);
	     reader->Update();

	     ImageType::Pointer image = reader->GetOutput();

	     ImageType::SizeType size; size = image->GetLargestPossibleRegion().GetSize();
	     for (int i=0; i < size[0]; i++)
	     {
	    	 for (int j=0; j< size[1]; j++)
	    	 {
	    		 for (int k=0; k < size[2]; k=k+2)
	    		 {
	    			 ImageType::IndexType index;
	    			 index[0]=i; index[1]=j; index[2]=k;
	    			 image->SetPixel(index, 0);
	    		 }
	    	 }
	     }


	      typedef itk::ImageFileWriter<ImageType> WriterType;
	      WriterType::Pointer writer = WriterType::New();

	      writer->SetFileName(output_Image_n);
	      writer->SetInput(image);
	      writer->Update();

	     return 0;
}


