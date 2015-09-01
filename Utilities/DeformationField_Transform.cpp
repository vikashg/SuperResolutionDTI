#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "../GetPot/GetPot"
#include "itkDiffusionTensor3D.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkMatrix.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_inverse.h"
#include "itkDisplacementFieldTransform.h"
#include "../inc/UnweightedLeastSquaresTensorFit.h"
#include "itkPoint.h"

using namespace std;

int main (int argc, char *argv[])
{

    GetPot cl (argc, const_cast<char**>(argv));
    if( cl.size() == 1 || cl.search (2,"--help","-h") )
    {
        std::cout << "Not Enough Arguments" << std::endl;
        std::cout << "Scales the tensors with a scalar factor" << std::endl;
        std::cout << "Usage: -trueB0 <true B0> -m <MaskImage> -true <True Tensors> -f <flag for extended gradient> -t <initial tensor estimate> -g <gradient> -o <Output File> -s <Sigma> -nm <Noise Model> -Sim <intelligent COnvergence>" << std::endl;
      return -1;
    }

	
	const string def_n = cl.follow("NoFile", 1, "-d");
	
	typedef itk::Point<float, 3> PointType;
	

	typedef itk::Vector<float, 3> VectorType;
	typedef itk::Image<VectorType, 3 > DisplacementImageType;
	typedef itk::ImageFileReader<DisplacementImageType> DisplacementImageReaderType;
	
	DisplacementImageReaderType::Pointer displaceReader = DisplacementImageReaderType::New();
	displaceReader->SetFileName(def_n);
	displaceReader->Update();
	
	DisplacementImageType::Pointer defField = displaceReader->GetOutput();

	typedef itk::DisplacementFieldTransform<float, 3> DisplacementFieldTransformType;		
	DisplacementImageType::IndexType testIndex;
	testIndex[0]=78; testIndex[1]=120; testIndex[2]=156;
	
	std::cout << defField->GetPixel(testIndex) << std::endl;

	PointType P1;
	P1[0]=150; P1[1]=150; P1[2]=50;

	typedef itk::VectorLinearInterpolateImageFunction<DisplacementImageType> VectorInterpolatorType;
	
	VectorInterpolatorType::Pointer vecInterp = VectorInterpolatorType::New();	
	vecInterp->SetInputImage(defField);	
	VectorType P = vecInterp->Evaluate(P1);
/*
	DisplacementFieldTransformType::Pointer filterTransform = DisplacementFieldTransformType::New();
	filterTransform->SetDisplacementField(defField);
	
	DisplacementImageType::Pointer temp = filterTransform->GetDisplacementField() ;
	
	std::cout << temp->GetPixel(testIndex) << std::endl;

	std::cout << filterTransform->TransformPoint(P1) << std::endl;	
*/


	return 0;
}
