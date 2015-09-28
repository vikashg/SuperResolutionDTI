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
#include "itkDiffusionTensor3D.h"
#include "CopyImage.h"
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
	const string maskImage_n  =  cl.follow("NoFile", 1, "-m");

	typedef itk::Image<float, 3> ImageType;
	typedef itk::ImageFileReader<ImageType> ReaderType;

	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName(maskImage_n.c_str());
	reader-> Update();
	ImageType::Pointer maskimage = reader->GetOutput();

 	typedef itk::ImageRegionIterator<ImageType> ImageIterator;
	ImageIterator it(maskimage, maskimage->GetLargestPossibleRegion());

	typedef itk::DiffusionTensor3D<float> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;
	typedef itk::ImageFileReader<TensorImageType> TensorImageReaderType;
	TensorImageReaderType::Pointer tensorReader = TensorImageReaderType::New();

	tensorReader->SetFileName(image_n.c_str());
	tensorReader->Update();
	TensorImageType::Pointer tensorImage = tensorReader->GetOutput();

	typedef itk::ImageRegionIterator<TensorImageType> TensorIterator;
	TensorIterator itT(tensorImage, tensorImage->GetLargestPossibleRegion());
	
	typedef DiffusionTensorType::EigenValuesArrayType EigenArrayType;

	ImageType::Pointer Eig1 = ImageType::New();
	ImageType::Pointer Eig2 = ImageType::New();
	ImageType::Pointer Eig3 = ImageType::New();

	CopyImage cpImage;
	cpImage.CopyScalarImage(maskimage, Eig1);
	cpImage.CopyScalarImage(maskimage, Eig2);
	cpImage.CopyScalarImage(maskimage, Eig3);

	for (it.GoToBegin(), itT.GoToBegin(); !it.IsAtEnd(), !itT.IsAtEnd(); ++it, ++itT)
	{
		if (it.Get() != 0)
		{
			EigenArrayType eig;
			DiffusionTensorType D = itT.Get();
			D.ComputeEigenValues(eig);
			
			Eig1->SetPixel(itT.GetIndex(), eig[0]);
			Eig2->SetPixel(itT.GetIndex(), eig[1]);
			Eig3->SetPixel(itT.GetIndex(), eig[2]);
		}
	}	


	typedef itk::ImageFileWriter<ImageType> WriterType;
	WriterType::Pointer writer1 = WriterType::New();
	writer1->SetFileName("Eig_1.nii.gz");	writer1->SetInput(Eig1); writer1->Update();
	WriterType::Pointer writer2 = WriterType::New();
	writer2->SetFileName("Eig_2.nii.gz");	writer2->SetInput(Eig2); writer2->Update();
	WriterType::Pointer writer3 = WriterType::New();
	writer3->SetFileName("Eig_3.nii.gz");	writer3->SetInput(Eig3); writer3->Update();

	return 0;
}
