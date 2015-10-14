
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
#include "JointTensorEstimation_LNLS.h"
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




     const string file_g_n = cl.follow("NoFile",1, "-g");
     const string fileIn  = cl.follow("NoFile",1,"-iLR");
     const string fileIn_HR  = cl.follow("NoFile",1,"-iHR");
     const string B0_n     =  cl.follow("NoFile", 1, "-B0_LR");
     const string mask_LR_n   = cl.follow("NoFile",1, "-mLR");
     const int numOfIter   = cl.follow(1,1, "-n");
     const float kappa_L   = cl.follow(0.05, 1, "-k");
     const float lambda_L  = cl.follow(0.25, 1, "-lamb_L");
     const string B0Image_HR_n = cl.follow("NoFile", 1, "-B0HR");
     const string dispField_n = cl.follow("NoFile",1,"-d");
     const string T1Image_n   = cl.follow("NoFile",1,"-T1");
     const string mask_HR_n = cl.follow("NoFile", 1, "-mHR");
     const string mask_uner_HR_n = cl.follow("NoFile",1,"-unerHR");
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

    //Read T1 image
	ScalarImageReaderType::Pointer scalarReader = ScalarImageReaderType::New();
	scalarReader->SetFileName(T1Image_n.c_str());
	scalarReader->Update();
	
	ScalarImageType::Pointer T1_image = scalarReader->GetOutput();

	std::cout << "Read T1 image " << std::endl;

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
	
	 std::cout << "Read Deformation Field" << std::endl;
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

	
	ScalarImageReaderType::Pointer unEroded_maskReader = ScalarImageReaderType::New();
	unEroded_maskReader->SetFileName(mask_uner_HR_n.c_str());
	unEroded_maskReader->Update();

	ScalarImageType::Pointer maskUnEroded = unEroded_maskReader->GetOutput();
	
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
    
	for (int i=0; i < numOfImagesHR  ; i++) // change of numOfImages
         {
             char filename[256];
             fileHR >> filename;
             ScalarImageReaderType::Pointer myReader=ScalarImageReaderType::New();
             myReader->SetFileName(filename);
	     myReader->Update();
             std::cout << "Reading.." << filename << std::endl; // add a try catch block
	    DWIList_HR.push_back(myReader->GetOutput());
	}

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

	// Compute the matrix
/*	MapFilterLR2HRDisp filter;
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

	ComposeImageFilter composeFilter;
	composeFilter.GetHRImage(T1_image);
	composeFilter.GetLRImage(B0_image_LR);
	composeFilter.ReadMatrix(MapHR2LR);

	ScalarImageType::Pointer tempImage1 = composeFilter.ComposeIt();


	ComposeImageFilter composeFilter2;
	composeFilter2.GetHRImage(B0_image_LR);
	composeFilter2.GetLRImage(T1_image);
	composeFilter2.ReadMatrix(MapLR2HR);
	
	ScalarImageType::Pointer tempImage2 = composeFilter2.ComposeIt();
	
	std::cout << "Composing done " << std::endl;
//
	typedef itk::ImageFileWriter<ScalarImageType> ScalarWriterType;
	ScalarWriterType::Pointer scalarWriter = ScalarWriterType::New();
	scalarWriter->SetFileName("TempImage2.nii.gz");
	scalarWriter->SetInput(tempImage2);
	scalarWriter->Update();

	ScalarWriterType::Pointer scalarWriter2 = ScalarWriterType::New();
	scalarWriter2->SetFileName("TempImage1.nii.gz");
	scalarWriter2->SetInput(tempImage1);
	scalarWriter2->Update();
*/
/*

	UnweightedLeastSquaresTensorEstimation UnWeightedTensorEstimator;
	UnWeightedTensorEstimator.ReadDWIList(DWIList_HR);
	UnWeightedTensorEstimator.ReadMask(maskUnEroded);
	UnWeightedTensorEstimator.ReadBVal(1.0);
        UnWeightedTensorEstimator.ReadGradientList(gradientImageList);
	UnWeightedTensorEstimator.ReadB0Image(B0Image_HR);	

	std::cout << "Computing Stupid Tensor " << std::endl;
	TensorImageType::Pointer tensorImage_init = UnWeightedTensorEstimator.Compute();

*/
	
/*	typedef itk::ImageFileReader<TensorImageType> TensorReaderType;
	TensorReaderType::Pointer tensorReader = TensorReaderType::New();
	tensorReader->SetFileName("TensorImage.nii.gz");
	tensorReader->Update();
	TensorImageType::Pointer tensorImage_init = tensorReader->GetOutput();
*/	
	TensorUtilities utilsTensors;

	//Correct the TensorImages
		// Replace NaNs
//		TensorImageType::Pointer tensorImage_removeNans = utilsTensors.ReplaceNaNsReverseEigenValue(tensorImage_init, maskImage_HR);
		
		//Compute the Log 
//		TensorImageType::Pointer log_tensorImage = utilsTensors.LogTensorImageFilter(tensorImage_removeNans, maskImage_HR);
		
		//Replace Nans&Infs
//		TensorImageType::Pointer removed_nans_logTensorImage = utilsTensors.ReplaceNaNsInfs(log_tensorImage, maskImage_HR);
		
		// Exp-ed Tensors
//		TensorImageType::Pointer padded_tensorImage_init = utilsTensors.ExpTensorImageFilter(removed_nans_logTensorImage, maskImage_HR);
	// 

/*
	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;	
	TensorWriterType::Pointer tensorWriter = TensorWriterType::New();
	tensorWriter->SetFileName("TensorImage.nii.gz");
	tensorWriter->SetInput(padded_tensorImage_init);
	tensorWriter->Update();
*/
		TensorImageType::IndexType IndexG, IndexB;
		IndexG[0]=123; IndexG[1]=138; IndexG[2]=171;
		
		IndexB[0]=123; IndexB[1]=159; IndexB[2]=201;

		typedef itk::ImageFileReader<TensorImageType> TensorReaderType;
		TensorReaderType::Pointer tensorReader = TensorReaderType::New();
		tensorReader->SetFileName("tensorImage.nii.gz");
		tensorReader->Update();
		
		TensorImageType::Pointer padded_tensorImage_init = tensorReader->GetOutput();
	//Compute Sigma 
		
//		std::cout << "Good Tensor " << padded_tensorImage_init->GetPixel(IndexG) << std::endl;
//		std::cout << " Bad Tensor " << padded_tensorImage_init->GetPixel(IndexB) << std::endl;
		

		vnl_vector<RealType> Sigma; 
		
		Sigma.set_size(DWIList.size());
		Sigma.fill(1.0);
/*		ComputeSigma_LR computeSigma;
		computeSigma.ReadB0Image_HR(B0Image_HR);
		computeSigma.ReadB0Image_LR(B0_image_LR);
		computeSigma.ReadBVal(1.0);
		computeSigma.ReadDWIList(DWIList);
		computeSigma.ReadMaskImage_HR(maskImage_HR);
		computeSigma.ReadGradientList(gradientImageList);
		computeSigma.ReadTensorImage(padded_tensorImage_init);
		computeSigma.ReadMapMatrix(MapHR2LR);
		computeSigma.ReadLRImage(maskImage_LR);

		Sigma  = computeSigma.ComputeAttenuation_woSR();
*/
		std::cout << Sigma << std::endl;

		// Compute Joint Estimation

/*		typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;
		TensorWriterType::Pointer tensorWriter = TensorWriterType::New();
		tensorWriter->SetFileName("TensorCorr.nii.gz");
		tensorWriter->SetInput(padded_tensorImage_init);
		tensorWriter->Update();
*/

		std::cout << "Sigma done " << std::endl;

//		TensorUtilities utilsTensor;
//		TensorImageType::Pointer removed_nans_logTensorImage = utilsTensor.LogTensorImageFilter(padded_tensorImage_init, maskImage_HR);


		JointTensorEstimation jTestimation;
		jTestimation.ReadDWIListHR(DWIList_HR);
		jTestimation.ReadDWIListLR(DWIList);

		jTestimation.ReadGradientList(gradientImageList);
		jTestimation.ReadInitialTensorImage(padded_tensorImage_init);

		jTestimation.ReadHRMask(maskImage_HR);
		jTestimation.ReadLRMask(maskImage_LR);

		jTestimation.ReadB0ImageHR(B0Image_HR);
		jTestimation.ReadB0ImageLR(B0_image_LR);

//		jTestimation.ReadMapMatrixLR2HR(MapLR2HR);
//		jTestimation.ReadMapMatrixHR2LR(MapHR2LR);
		jTestimation.ReadBVal(1.0);

		jTestimation.ReadStepSize(1000);
	
		jTestimation.ReadSigma(Sigma);
		jTestimation.ReadKappa(0.25);
		jTestimation.ReadLambda(10);
		jTestimation.ReadNumOfIterations(20);
		//Testing Components
		ImageListType testDiffImageList, AttenuationList;

		std::cout << "Read everything " << std::endl;
		TensorImageType::Pointer tensorImage = jTestimation.UpdateTerms1();
	
		//AttenuationList = jTestimation.ComputeAttenuation(removed_nans_logTensorImage);


//		typedef itk::ImageFileWriter<TensorImageType> TensorImageWriterType;
//		TensorImageWriterType::Pointer tensorWriter1 = TensorImageWriterType::New();
//		tensorWriter1->SetFileName("EstimatedTensor_Sim.nii.gz");
//		tensorWriter1->SetInput(tensorImage);
//		tensorWriter1->Update();

	
        return 0;

}

