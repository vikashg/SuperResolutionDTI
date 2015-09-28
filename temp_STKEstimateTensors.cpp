
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
#include "ComputeSigma.h"
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




     const string file_g_n = cl.follow("NoFile",1, "-g");
     const string fileIn  = cl.follow("NoFile",1,"-i");
     const string fileIn_HR  = cl.follow("NoFile",1,"-iHR");
     const string B0_n     =  cl.follow("NoFile", 1, "-B0");
     const string mask_LR_n   = cl.follow("NoFile",1, "-mLR");
     const int numOfIter   = cl.follow(1,1, "-n");
     const float kappa_L   = cl.follow(0.05, 1, "-k");
     const float lambda_L  = cl.follow(0.25, 1, "-lamb_L");
     const string B0Image_HR_n = cl.follow("NoFile", 1, "-B0HR");
     const string dispField_n = cl.follow("NoFile",1,"-d");
     const string T1Image_n   = cl.follow("NoFile",1,"-T1");
     const string mask_HR_n = cl.follow("NoFile", 1, "-mHR");
    // Usual Typedefs
    typedef float RealType;
    const int ImageDim  =3;

    typedef itk::Image<RealType, ImageDim> ScalarImageType;
    typedef itk::Vector<double, ImageDim> VectorType;
    typedef itk::Image<VectorType, ImageDim> DeformationFieldType;
    typedef itk::Image<VectorType, ImageDim> VectorImageType;

    typedef itk::ImageFileReader<ScalarImageType> ScalarImageReaderType;
    typedef itk::ImageFileWriter<ScalarImageType> ScalarImageWriterType;

    //Read T1 image
	ScalarImageReaderType::Pointer scalarReader = ScalarImageReaderType::New();
	scalarReader->SetFileName(T1Image_n.c_str());
	scalarReader->Update();
	
	ScalarImageType::Pointer T1_image = scalarReader->GetOutput();

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

	// Read deformation field
	typedef itk::ImageFileReader<DeformationFieldType> DeformationFieldReaderType;
	DeformationFieldReaderType::Pointer deformationFieldReader = DeformationFieldReaderType::New();

	deformationFieldReader->SetFileName(dispField_n.c_str());
	deformationFieldReader->Update();
	
	DeformationFieldType::Pointer defField = deformationFieldReader->GetOutput();
	

	// Read Mask Image Spatial
	typedef itk::ImageMaskSpatialObject<ImageDim> MaskSpatialObjectType;
	typedef MaskSpatialObjectType::ImageType MaskSpatialImageType;
	typedef itk::ImageFileReader<MaskSpatialImageType> MaskSpatialImageReaderType;	

	MaskSpatialImageReaderType::Pointer spatialReader = MaskSpatialImageReaderType::New();
	spatialReader->SetFileName(mask_LR_n.c_str());
	spatialReader->Update();
	
	MaskSpatialImageType::Pointer maskSpatialImage_LR = spatialReader->GetOutput();

	//Read Mask Image Normal
	ScalarImageReaderType::Pointer maskImageReader = ScalarImageReaderType::New();
	maskImageReader->SetFileName(mask_LR_n.c_str());
	maskImageReader->Update();
	
	ScalarImageType::Pointer maskImage_LR = maskImageReader->GetOutput();

		
	// Resample diffusion Images	
/*	typedef itk::WarpImageFilter<ScalarImageType, ScalarImageType, DeformationFieldType> WarpImageFilterType;
	typedef itk::ImageFileWriter<ScalarImageType> ScalarImageWriterType;
	
	

	for (int i =0; i < numOfImages; i++)
	{
		WarpImageFilterType::Pointer warpImageFilter = WarpImageFilterType::New();
		warpImageFilter->SetOutputSpacing(T1_image->GetSpacing());
		warpImageFilter->SetOutputOrigin(T1_image->GetOrigin());
		warpImageFilter->SetDisplacementField(defField);
		warpImageFilter->SetInput(DWIList[i]);
		warpImageFilter->Update();		
		
		ScalarImageType::Pointer imageDWI = warpImageFilter->GetOutput();	
		
		std::ostringstream num_con;
		num_con << i ;
		std::string result = num_con.str() + ".nii.gz";
		
		ScalarImageWriterType::Pointer writer = ScalarImageWriterType::New();
		writer->SetFileName(result);
		writer->SetInput(imageDWI);
		writer->Update();
		std::cout<< i << "Done" << std::endl;
	}*/

	// Read Gradient List ScalarValues
		typedef itk::Vector<double, 3> VectorDoubleType;
		typedef std::vector<VectorDoubleType> GradientListType;
		GradientListType GradientList;
		std::ifstream fileg(file_g_n.c_str());
		
		int numOfGrads =0;
		fileg >> numOfGrads;
		for (int i=0; i < numOfGrads ; i++)
		{
		VectorType g;
		fileg >> g[0]; fileg >> g[1]; fileg >> g[2];
		GradientList.push_back(g);
		}	

	// Transform gradients
	

	// Read the HR diffusion images, 
	
	typedef std::vector<ScalarImageType::Pointer> ImageListType;
	ImageListType DWIList_HR;
	
	std::ifstream fileHR(fileIn_HR.c_str());
        int numOfImagesHR = 0;
        fileHR >> numOfImagesHR;	
    
/*	for (int i=0; i < numOfImagesHR  ; i++) // change of numOfImages
         {
             char filename[256];
             fileHR >> filename;
             ScalarImageReaderType::Pointer myReader=ScalarImageReaderType::New();
             myReader->SetFileName(filename);
	     myReader->Update();
             std::cout << "Reading.." << filename << std::endl; // add a try catch block
	    DWIList_HR.push_back(myReader->GetOutput());
	}
*/
		// Read HR mask image spatial
	MaskSpatialImageReaderType::Pointer spatialReader_HR = MaskSpatialImageReaderType::New();
	spatialReader_HR->SetFileName(mask_HR_n.c_str());
	spatialReader_HR->Update();
    MaskSpatialImageType::Pointer maskSpatial_HR = spatialReader_HR->GetOutput();

		//Read HR mask image Normal
	ScalarImageReaderType::Pointer maskImage_HR_reader = ScalarImageReaderType::New();
	maskImage_HR_reader->SetFileName(mask_HR_n.c_str());
	maskImage_HR_reader->Update();
	ScalarImageType::Pointer maskImage_HR = maskImage_HR_reader->GetOutput();
		// Read B0 HR image
	ScalarImageReaderType::Pointer B0Image_HR_reader = ScalarImageReaderType::New();
	B0Image_HR_reader->SetFileName(B0Image_HR_n.c_str());
	B0Image_HR_reader->Update();
	ScalarImageType::Pointer B0Image_HR = B0Image_HR_reader->GetOutput();	

		//Read B0 LR image
	ScalarImageReaderType::Pointer B0_image_LR_reader = ScalarImageReaderType::New();
	B0_image_LR_reader->SetFileName(B0_n.c_str());
	B0_image_LR_reader->Update();
	ScalarImageType::Pointer B0_image_LR = B0_image_LR_reader->GetOutput();

	// ComputeGradientImages
/*	 TransformGradients transformGradients;
	 transformGradients.ReadMaskImage(maskImage_HR);
	 transformGradients.ReadDeformationField(defField);
	 transformGradients.ReadGradients(GradientList);
	 transformGradients.ComputeGradients();
	 
	typedef std::vector<VectorImageType::Pointer> GradientImageListType;
	GradientImageListType GradientImageList;
	GradientImageList = transformGradients.GetGradientImages();

	std::cout << "Transformed all Gradients... Done." << std::endl;
	
	typedef itk::ImageFileWriter<VectorImageType> GradientImageWriterType;
	for (int i=0; i < numOfGrads; i++)
	{
	GradientImageWriterType::Pointer gradientImageWriter = GradientImageWriterType::New();
	
		std::ostringstream c; 
		c << i; 	
		std::string tempName;
		tempName = "Gradient_" + c.str()  + ".nii.gz";
		
		gradientImageWriter->SetFileName(tempName);
		gradientImageWriter->SetInput(GradientImageList[i]);
		gradientImageWriter->Update();

	}
*/

	//Read GradientImages
	const string file_gradImage_n = cl.follow("NoFile", 1, "-fG");
	std::ifstream fileGImg(file_gradImage_n.c_str());
	
	int numOfGradImages = 0;
	fileGImg >> numOfGradImages;
	
	typedef itk::ImageFileReader<VectorImageType> GradientImageReaderType;
	typedef std::vector<VectorImageType::Pointer> GradientImageListType;
	GradientImageListType gradientImageList;	

/*	for (int i=0; i < numOfGradImages; i++)
	{
		char filename[25];
		fileGImg >> filename;
		VectorImageType::Pointer gradientImage = VectorImageType::New();
		GradientImageReaderType::Pointer gradientImageReader = GradientImageReaderType::New();
		gradientImageReader->SetFileName(filename);
		gradientImageReader->Update();
		gradientImageList.push_back(gradientImageReader->GetOutput()) ;

		std::cout << "Reading...." << filename << std::endl;		
	
	}*/

	// Compute the matrix
	MapFilterLR2HRDisp filter;
	filter.ReadFixedImage(T1_image); //	
	filter.ReadMovingImage(B0_image_LR); 
	filter.ReadDeformationField(defField);
	filter.ReadMaskImage(maskSpatial_HR);
	filter.ComputeMapWithDefField();

	vnl_sparse_matrix<float> MapLR2HR, MapHR2LR;
	
	MapLR2HR = filter.GetLR2HRMatrix();
	MapHR2LR = filter.GetHR2LRMatrix();

	std::cout << "Computing Map done " << std::endl;

	std::cout << B0_image_LR->GetLargestPossibleRegion().GetSize() << std::endl;
//
//	ComposeImageFilter composeFilter;
//	composeFilter.GetHRImage(T1_image);
//	composeFilter.GetLRImage(B0_image_LR);
//	composeFilter.ReadMatrix(MapLR2HR);
//
//	ScalarImageType::Pointer tempImage1 = composeFilter.ComposeIt();
	
//	std::cout << "Composing done " << std::endl;
//
//	typedef itk::ImageFileWriter<ScalarImageType> ScalarWriterType;
//	ScalarWriterType::Pointer scalarWriter = ScalarWriterType::New();
//	scalarWriter->SetFileName("TempImage.nii.gz");
//	scalarWriter->SetInput(tempImage1);
//	scalarWriter->Update();

/*	UnweightedLeastSquaresTensorEstimation UnWeightedTensorEstimator;
	UnWeightedTensorEstimator.ReadDWIList(DWIList_HR);
	UnWeightedTensorEstimator.ReadMask(maskImage_HR);
	UnWeightedTensorEstimator.ReadBVal(1.0);
        UnWeightedTensorEstimator.ReadGradientList(gradientImageList);
	UnWeightedTensorEstimator.ReadB0Image(B0Image_HR);	

	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, ImageDim> TensorImageType;
	TensorImageType::Pointer tensorImage_init = UnWeightedTensorEstimator.Compute();

	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;	
	TensorWriterType::Pointer tensorWriter = TensorWriterType::New();
	tensorWriter->SetFileName("TensorImage.nii.gz");
	tensorWriter->SetInput(tensorImage_init);
	tensorWriter->Update();
	
	//Compute Sigma 
		vnl_vector<RealType> Sigma; 
		 
	
	// Compute Joint Estimation 
//	JointTensorEstimation jTestimation;
//	jTestimation.ReadBVal(1.0);
//	jTestimation.ReadDWIList(DWIList_HR);
//	jTestimation.ReadMaskImage(maskImage_HR);
//	jTestimation.ReadB0Image(B0Image_HR);
//	jTestimation.ReadGradientList(gradientImageList);
//	jTestimation.ReadInitialTensorImage(tensorImage_init);
//	jTestimation.ReadKappa(kappa_L);
*/	
		
	
   return 0;

}






