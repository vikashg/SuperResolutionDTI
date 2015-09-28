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

	const string image_n = cl.follow("NoFile", 1, "-i");

	typedef itk::Image<float, 3> ImageType;
	typedef itk::ImageFileReader<ImageType> ReaderType;

	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName(image_n.c_str());
	reader-> Update();
	ImageType::Pointer image = reader->GetOutput();

 	typedef itk::ImageRegionIterator<ImageType> ImageIterator;
	ImageIterator it(image, image->GetLargestPossibleRegion());

	for (it.GoToBegin(); !it.IsAtEnd(); ++it)
	{
	 	if (it.Get() < -5 )
		{
		 it.Set(1.0);
		  std::cout << it.GetIndex() << std::endl;
		}
		else if (it.Get() > 5 )
		{
		 it.Set(1.0);
		  std::cout << it.GetIndex() << std::endl;
		}
		else 
		{
			it.Set(0.0);
		}
	}	

	typedef itk::ImageFileWriter<ImageType> WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName("Trace_mask.nii.gz");
	writer->SetInput(image);
	writer->Update();

	return 0;
}
