/*
 * PropagateTransformGradients.cpp
 *
 *  Created on: Aug 18, 2015
 *      Author: vgupta
 */



#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "../GetPot/GetPot"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "../inc/itkDeformationFieldGradientTensorImageFilter.h"
#include "itkImageFileWriter.h"

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

	    const string Grad_n = cl.follow("NoFile",1,"-i");
	    typedef itk::Matrix<float, 3, 3> MatrixType;
		typedef itk::Image<MatrixType, 3> MatrixImageType;
		typedef itk::Image<itk::Vector<float, 3>, 3> VectorImageType;

		typedef itk::ImageFileReader<VectorImageType> ReaderType;
		ReaderType::Pointer reader = ReaderType::New();

		reader->SetFileName(Grad_n);
		reader->Update();

		VectorImageType::Pointer vectorImage = reader->GetOutput();

		VectorImageType::IndexType testIndex;
		testIndex[0]=70; testIndex[1]=119 ; testIndex[2]=128;

		std::cout << vectorImage->GetPixel(testIndex) << std::endl;
	    return 0;
	}
