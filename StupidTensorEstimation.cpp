
/*
 * STKEstimateTensors_DispField.cpp
 *
 *  Created on: Sep 3, 2015
 *      Author: vgupta
 */

#include "TransformGradients.h"
#include "iostream"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "GetPot/GetPot"
#include "itkDiffusionTensor3D.h"
#include "JointTensorEstimation.h"
#include "CopyImage.h"
#include "../inc/UnweightedLeastSquaresTensorFit.h"
#include "ComputeSigma_LR.h"
#include "itkImageRegionIterator.h"
#include "TotalEnergy.h"
#include "vnl/vnl_matrix_exp.h"
#include "itkDisplacementFieldTransform.h"
#include "itkImageMaskSpatialObject.h"
#include "MapFilterLR2HRDispField.h"
#include "ComposeImage.h"
#include "vnl/vnl_sparse_matrix.h"
#include "itkResampleImageFilter.h"
#include "itkWarpImageFilter.h"
#include "TensorUtilites.h"

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

     const string fileIn  = cl.follow("NoFile",1,"-iLR");
     const string B0_n     =  cl.follow("NoFile", 1, "-B0_LR");
     const string mask_LR_n   = cl.follow("NoFile",1, "-mLR");
    // Usual Typedefs
    typedef float RealType;
    const int ImageDim  =3;

    typedef itk::Image<RealType, ImageDim> ScalarImageType;
    typedef itk::Vector<double, ImageDim> VectorType;
    typedef itk::Image<VectorType, ImageDim> DeformationFieldType;
    typedef itk::Image<VectorType, ImageDim> VectorImageType;

    typedef itk::ImageFileReader<ScalarImageType> ScalarImageReaderType;
    typedef itk::ImageFileWriter<ScalarImageType> ScalarImageWriterType;

    typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
    typedef itk::Image<DiffusionTensorType, ImageDim> TensorImageType;

    // Read LR ImageList
	typedef std::vector<ScalarImageType::Pointer> ImageListType;
	ImageListType DWIList;
	
	std::ifstream file(fileIn.c_str());
        int numOfImages = 0;
        file >> numOfImages;	
    
	for (int i=0; i < numOfImages  ; i++) // change of numOfImages
         {
             char filename[256];
             file >> filename;
             ScalarImageReaderType::Pointer myReader=ScalarImageReaderType::New();
             myReader->SetFileName(filename);
             std::cout << "Reading.." << filename << std::endl; // add a try catch block
             myReader->Update();
             DWIList.push_back( myReader->GetOutput() ); //using push back to create a stack of diffusion images
         }

	

	
       //Read GradientImages
        const string file_gradImage_n = cl.follow("NoFile", 1, "-fG");
        std::ifstream fileGImg(file_gradImage_n.c_str());
        int numOfGradImages = 0;
	fileGImg >> numOfGradImages;

         typedef itk::ImageFileReader<VectorImageType> GradientImageReaderType;
         typedef std::vector<VectorImageType::Pointer> GradientImageListType;
         GradientImageListType gradientImageList;

         for (int i=0; i < numOfGradImages; i++)
         {
                 char filename[25];
                 fileGImg >> filename;
                 VectorImageType::Pointer gradientImage = VectorImageType::New();
                 GradientImageReaderType::Pointer gradientImageReader = GradientImageReaderType::New();
                 gradientImageReader->SetFileName(filename);
                 gradientImageReader->Update();
                 gradientImageList.push_back(gradientImageReader->GetOutput()) ;

                 std::cout << "Reading...." << filename << std::endl;

         }


	//Read Mask Image Normal
	ScalarImageReaderType::Pointer maskImageReader = ScalarImageReaderType::New();
	maskImageReader->SetFileName(mask_LR_n.c_str());
	maskImageReader->Update();
	
	ScalarImageType::Pointer maskImage_LR = maskImageReader->GetOutput();

	//Read B0 Image 
	ScalarImageReaderType::Pointer B0ImageReader = ScalarImageReaderType::New();
	B0ImageReader->SetFileName(B0_n.c_str());
	B0ImageReader->Update();
	ScalarImageType::Pointer B0Image_LR = B0ImageReader->GetOutput();


	UnweightedLeastSquaresTensorEstimation UnWeightedTensorEstimator;
	UnWeightedTensorEstimator.ReadDWIList(DWIList);
	UnWeightedTensorEstimator.ReadMask(maskImage_LR);
	UnWeightedTensorEstimator.ReadBVal(1.0);
        UnWeightedTensorEstimator.ReadGradientList(gradientImageList);
	UnWeightedTensorEstimator.ReadB0Image(B0Image_LR);	

	std::cout << "Computing Stupid Tensor " << std::endl;
	TensorImageType::Pointer tensorImage_init = UnWeightedTensorEstimator.Compute();

	
	TensorUtilities utilsTensors;

	//Correct the TensorImages
		// Replace NaNs
		TensorImageType::Pointer tensorImage_removeNans = utilsTensors.ReplaceNaNsReverseEigenValue(tensorImage_init, maskImage_LR);
		
		//Compute the Log 
		TensorImageType::Pointer log_tensorImage = utilsTensors.LogTensorImageFilter(tensorImage_removeNans, maskImage_LR);
		
		//Replace Nans&Infs
		TensorImageType::Pointer removed_nans_logTensorImage = utilsTensors.ReplaceNaNsInfs(log_tensorImage, maskImage_LR);
		
		// Exp-ed Tensors
		TensorImageType::Pointer padded_tensorImage_init = utilsTensors.ExpTensorImageFilter(removed_nans_logTensorImage, maskImage_LR);
	// 

	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;	
	TensorWriterType::Pointer tensorWriter = TensorWriterType::New();
	tensorWriter->SetFileName("TensorImage.nii.gz");
	tensorWriter->SetInput(padded_tensorImage_init);
	tensorWriter->Update();

	
        return 0;

}

