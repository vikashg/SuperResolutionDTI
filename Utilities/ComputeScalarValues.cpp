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
#include "../inc/TensorUtilites.h"
#include "CopyImage.h"

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
	   const string mask_n = cl.follow("NoFile", 1, "-m");	 
	   const string out_n   = cl.follow("NoFile",1, "-o");

	    typedef itk::DiffusionTensor3D<float> DiffusionTensorType;
	    typedef itk::Image<DiffusionTensorType, 3> TensorImageType;
	    typedef itk::ImageFileReader<TensorImageType> TensorReaderType;
	    TensorReaderType::Pointer reader = TensorReaderType::New();

	    reader->SetFileName(image_n.c_str());
	    reader->Update();

	    TensorImageType::Pointer image = reader->GetOutput();


	    typedef itk::Image<float, 3> ScalarImageType;
	    typedef itk::ImageFileReader<ScalarImageType> ScalarReaderType;

	    ScalarReaderType::Pointer scalarReader = ScalarReaderType::New();
	    scalarReader->SetFileName(mask_n.c_str());
	    scalarReader->Update();
	    ScalarImageType::Pointer maskImage = scalarReader->GetOutput();
	
	    typedef itk::ImageRegionIterator<TensorImageType> TensorIterator;

	    typedef itk::ImageRegionIterator<ScalarImageType> ScalarIterator;
	    ScalarIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());
	

	    TensorUtilities  utilsTensor;
	    TensorImageType::Pointer logTensorImage = utilsTensor.LogTensorImageFilter(image, maskImage);


	    typedef itk::ImageFileWriter<TensorImageType> TensorImageWriterType;
	    TensorImageWriterType::Pointer tensorImageWriter = TensorImageWriterType::New();

            tensorImageWriter->SetFileName("LogTensorImage_stupid.nii.gz");
	    tensorImageWriter->SetInput(logTensorImage);
	    tensorImageWriter->Update();
	
	    std::ofstream  file;
	    file.open(out_n);

	 	ScalarImageType::Pointer TraceImage = ScalarImageType::New();
	    CopyImage	cpImage;
	    cpImage.CopyScalarImage(maskImage, TraceImage);

	    ScalarIterator itTr(TraceImage, TraceImage->GetLargestPossibleRegion());



	    TensorIterator itImg(logTensorImage, logTensorImage->GetLargestPossibleRegion());
	    for(itImg.GoToBegin(), itMask.GoToBegin(), itTr.GoToBegin(); !itTr.IsAtEnd(), !itImg.IsAtEnd(), !itMask.IsAtEnd(); ++itTr, ++itImg, ++itMask)
	    {
	    		file << itImg.Get().GetTrace() << std::endl;
	    		itTr.Set(itImg.Get().GetTrace()) ;
		

	   }


		typedef itk::ImageFileWriter<ScalarImageType> WriterType;
		WriterType::Pointer writer = WriterType::New();
	
		writer->SetFileName("LogTense_Trace.nii.gz");
		writer->SetInput(TraceImage);
		writer->Update();


	    file.close();


		return 0;
	}
