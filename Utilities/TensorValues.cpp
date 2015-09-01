/*
 * TensorValues.cxx
 *
 *  Created on: Jul 26, 2015
 *      Author: vgupta
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "../GetPot/GetPot"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "../inc/itkDeformationFieldGradientTensorImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkDiffusionTensor3D.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"

#include "itkVector.h"
using namespace std;
int main (int argc, char *argv[])
	{
	const unsigned int ImageDimension = 2;
	    GetPot cl (argc, const_cast<char**>(argv));
	    if( cl.size() == 1 || cl.search (2,"--help","-h") )
	    {
	        std::cout << "Not Enough Arguments" << std::endl;
	        std::cout << "Generate the Gradient Table" << std::endl;
	        std::cout << "Usage:  return -1" << std::endl;
	    }

	    const string image_n = cl.follow("NoFile",1, "-i");
	    const string out_n   = cl.follow("NoFile",1, "-o");

	    typedef itk::DiffusionTensor3D<float> DiffusionTensorType;
	    typedef itk::Image<DiffusionTensorType, 3> ImageType;
	    typedef itk::ImageFileReader<ImageType> ReaderType;
	    ReaderType::Pointer reader = ReaderType::New();

	    reader->SetFileName(image_n);
	    reader->Update();

	    ImageType::Pointer image = reader->GetOutput();

	    typedef itk::ImageRegionIterator<ImageType> TensorIterator;
	    TensorIterator itImg(image, image->GetLargestPossibleRegion());

	    std::ofstream  file;
	    file.open(out_n);

	    for(itImg.GoToBegin(); !itImg.IsAtEnd(); ++itImg)
	    {
	    	file << itImg.Get() << std::endl;
	    }

	    file.close();



		return 0;
	}



