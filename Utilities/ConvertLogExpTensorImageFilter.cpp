
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "CopyImage.h"
#include "../GetPot/GetPot"
#include "TensorUtilites.h"

using namespace std;

int main(int argc, char *argv[])
{
	
	GetPot cl(argc, const_cast<char**>(argv));
	
	const string tensorImage_n = cl.follow("NoFile", 1, "-i");
	const string outImage_n = cl.follow("NoFile",1, "-o");
	const string maskImage_n = cl.follow("NoFile",1,"-m");

	typedef itk::DiffusionTensor3D<float> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;
	typedef itk::ImageFileReader<TensorImageType> TensorReaderType;
	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;

	TensorReaderType::Pointer tensorReader  = TensorReaderType::New();
	tensorReader->SetFileName(tensorImage_n.c_str());
	tensorReader->Update();
	
	TensorImageType::Pointer tensorImage = tensorReader->GetOutput();
	
	typedef itk::Image<float, 3> ScalarImageType;
	typedef itk::ImageFileReader<ScalarImageType> ScalarImageReaderType;
	ScalarImageReaderType::Pointer scalarImageReader = ScalarImageReaderType::New();
	
	scalarImageReader->SetFileName(maskImage_n.c_str());
	scalarImageReader->Update();
	ScalarImageType::Pointer maskImage = scalarImageReader->GetOutput();

	TensorUtilities utilsTensor;
	TensorImageType::Pointer tensor_replace_Nans = utilsTensor.ReplaceNaNsReverseEigenValue(tensorImage, maskImage);

	TensorImageType::Pointer log_tensorImage = utilsTensor.LogTensorImageFilter(tensor_replace_Nans, maskImage);
	
	TensorImageType::Pointer removed_nans_logTensorImage= utilsTensor.ReplaceNaNsInfs(log_tensorImage, maskImage);

	typedef itk::ImageRegionIterator<TensorImageType> TensorIterator;
	typedef itk::ImageRegionIterator<ScalarImageType> ScalarIterator;

	TensorIterator itTens(removed_nans_logTensorImage, removed_nans_logTensorImage->GetLargestPossibleRegion());
	ScalarIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());

	for (itMask.GoToBegin(), itTens.GoToBegin(); !itMask.IsAtEnd(), !itTens.IsAtEnd();
		++itMask, ++itTens)
	{
	if (itMask.Get() !=0)
	{
		float Trace = itTens.Get().GetTrace();
		if ( (isnan(Trace) == 1 ) || (isinf(Trace) == 1) )
		{
		 std::cout << itMask.GetIndex() << std::endl;
		}
	}
	
	}

	std::cout << "Checking done " << std::endl;
				

	TensorImageType::Pointer expedTensor = utilsTensor.ExpTensorImageFilter(removed_nans_logTensorImage, maskImage);

	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;
	TensorWriterType::Pointer tensorWriter = TensorWriterType::New();
	tensorWriter->SetFileName(outImage_n.c_str());
	tensorWriter->SetInput(expedTensor);
	tensorWriter->Update();
	
	return 0;
}
