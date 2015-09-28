#include "iostream"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "../inc/TensorUtilites.h"
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

	TensorImageType::Pointer tensorProb = TensorImageType::New();
	cpImage.CopyTensorImage(tensorImage, tensorProb);

	DiffusionTensorType D_Identity;
	
	vnl_matrix<float> ZeroD_mat; ZeroD_mat.set_size(3,3);
 	ZeroD_mat.set_identity();

	TensorUtilities utils;
	D_Identity = utils.ConvertMat2DT(ZeroD_mat);

	//tensorProb->FillBuffer(D_Identity);
	
	for (it.GoToBegin(), itT.GoToBegin(); !it.IsAtEnd(), !itT.IsAtEnd(); ++it, ++itT)
	{
		if (it.Get() != 0)
		{
			EigenArrayType eig;
			DiffusionTensorType D = itT.Get();
			D.ComputeEigenValues(eig);
			
			if ( (eig[0] < 0) || (eig[1] < 0 ) || (eig[2] < 0) || eig[0] > 5 || (eig[1] > 5) || (eig[2] > 5))
			{
				tensorProb->SetPixel(itT.GetIndex(), D);	
				std::cout << eig << std::endl;	
			}			
			
		
		}
	}	

	typedef itk::ImageFileWriter<TensorImageType> TensorImageWriter;
	TensorImageWriter::Pointer tensorWriter = TensorImageWriter::New();
	tensorWriter->SetFileName("TensorImage_Prob.nii.gz");
	tensorWriter->SetInput(tensorProb);
	tensorWriter->Update();
	
	return 0;
}
