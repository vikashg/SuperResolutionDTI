#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "../GetPot/GetPot"
#include "itkImageRegionIterator.h"

using namespace std;

int main (int argc, char *argv[])
{

	GetPot cl (argc, const_cast<char**>(argv));
	const string image_n = cl.follow("NoFile",1, "-i");
	const string mask_n = cl.follow("NoFile", 1, "-m");
	
	typedef itk::Image<float, 3> ImageType;
	typedef itk::ImageFileReader<ImageType> ReaderType;

	ReaderType::Pointer reader_B0 = ReaderType::New();
	reader_B0->SetFileName(image_n.c_str());
	reader_B0->Update();

	ImageType::Pointer B0Image = reader_B0->GetOutput();

	ReaderType::Pointer reader_m = ReaderType::New();
	reader_m->SetFileName(mask_n.c_str());
	reader_m->Update();
	
	ImageType::Pointer maskImage = reader_m->GetOutput();

	typedef itk::ImageRegionIterator<ImageType> ImageRegionIterator;
	ImageRegionIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());
	ImageRegionIterator itB0(B0Image, B0Image->GetLargestPossibleRegion());

	float B0_thres =200;

	for (itMask.GoToBegin(), itB0.GoToBegin(); !itMask.IsAtEnd(), !itB0.IsAtEnd(); ++itMask, ++itB0)
	{
	if (itMask.Get() != 0)
	{
	 if (itB0.Get() <= B0_thres)
	  {
		itMask.Set(0);
	  } 
	}
	
	}	
	
	typedef itk::ImageFileWriter<ImageType> WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName("MaskThres.nii.gz");
	writer->SetInput(maskImage);
	writer->Update();


	return 0;
}
