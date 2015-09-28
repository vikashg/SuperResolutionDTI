/*
 * TestMappingDispField.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: vgupta
 */
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageMaskSpatialObject.h"
#include "../GetPot/GetPot"
#include "ComposeImage.h"
#include "UnweightedLeastSquaresTensorFit.h"
#include "ComputeSigma_LR.h"
#include <MapFilterLR2HRDispField.h>

#include <fstream>

#include <iostream>

using namespace std;
int main(int argc, char* *argv)
{


	     GetPot   cl(argc, argv);
	     if( cl.size() == 1 || cl.search(2, "--help", "-h") )
	      {
	          std::cout << "Not enough arguments" << std::endl;
	          return -1;
	      }

             bool flag=0;
	    vnl_sparse_matrix<float> MapLR2HR, MapHR2LR; //inverse mappings
	     typedef float RealType;
	     const int ImageDim =3;
	     typedef itk::Vector<double, 3> VectorType;
	     typedef itk::Image<VectorType, 3> DeformationFieldType;
	     typedef itk::Image<RealType , 3> ScalarImageType;
	     typedef itk::ImageFileReader<ScalarImageType> ReaderType;

	    ofstream myFile ("data.bin", ios::binary);



	  if (flag == 0 )
	   {  const string T1Image_n = cl.follow("NoFile", 1, "-T1");
	     const string B0ImageLR_n = cl.follow("NoFile",1, "-B0_LR");
	     const string dispField_n = cl.follow("NoFile", 1, "-d");
	     const string maskImage_HR_n = cl.follow("NoFile",1, "-mHR");

	     typedef itk::ImageFileReader<DeformationFieldType> DisplacementFileReaderType;
	     DisplacementFileReaderType::Pointer deformationFieldReader = DisplacementFileReaderType::New();

	     deformationFieldReader->SetFileName(dispField_n.c_str());
	     deformationFieldReader->Update();

	     DeformationFieldType::Pointer defField = deformationFieldReader->GetOutput();

	     typedef itk::ImageMaskSpatialObject<3> MaskSpatiolObjectType;
	     typedef MaskSpatiolObjectType::ImageType MaskSpatialImageType;
	     typedef itk::ImageFileReader<MaskSpatialImageType> SpatialImageReaderType;
	     SpatialImageReaderType::Pointer readerSpatial = SpatialImageReaderType::New();
	     readerSpatial->SetFileName(maskImage_HR_n.c_str());
	     readerSpatial->Update();

	     MaskSpatialImageType::Pointer maskImage_HR_spatial = readerSpatial->GetOutput();

	     ReaderType::Pointer T1_image_reader = ReaderType::New();

	     T1_image_reader->SetFileName(T1Image_n.c_str());
	     T1_image_reader->Update();
	     ScalarImageType::Pointer T1 = T1_image_reader->GetOutput();

	     ReaderType::Pointer B0_image_reader = ReaderType::New();
	     B0_image_reader->SetFileName(B0ImageLR_n.c_str());
	     B0_image_reader->Update();
	     ScalarImageType::Pointer B0_LR = B0_image_reader->GetOutput();

	     MapFilterLR2HRDisp filter;
	     filter.ReadFixedImage(T1); //T1
	     filter.ReadMovingImage(B0_LR); //B0
	     filter.ReadDeformationField(defField); // T1---> B0
	     filter.ReadMaskImage(maskImage_HR_spatial);
	     filter.ComputeMapWithDefField();

	     std::cout << "Computed the Map " << std::endl;

		myFile.write((char*)&MapHR2LR, sizeof(MapHR2LR));
	 	MapLR2HR = filter.GetLR2HRMatrix();
	 	MapHR2LR = filter.GetHR2LRMatrix();

		ComposeImageFilter composeFilter;
	 	composeFilter.GetHRImage(T1);
	 	composeFilter.GetLRImage(B0_LR);
	 	composeFilter.ReadMatrix(MapHR2LR);
		ScalarImageType::Pointer tempImage = composeFilter.ComposeIt();
		flag =1;



}

	
std::cout << "Flag " << flag << std::endl;	

	if (flag ==1 )
{		
	 	// Mapping Done
	 	//Read other stuff
	 	// Read HR DWIS
/*	     typedef std::vector<ScalarImageType::Pointer> ImageListType;
	     typedef itk::ImageFileReader<ScalarImageType> ScalarImageReaderType;
	     const string fileIn_HR  = cl.follow("NoFile",1,"-iHR");

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

	     //Read HR B0
	     const string B0_HR_n = cl.follow("NoFile",1,"-B0HR");
	     ScalarImageReaderType::Pointer reader_B0_HR = ScalarImageReaderType::New();
	     reader_B0_HR->SetFileName(B0_HR_n.c_str());
	     reader_B0_HR->Update();
	     ScalarImageType::Pointer B0_HR = reader_B0_HR->GetOutput();
*/
	     // Read HR mask normally
/*	     ScalarImageReaderType::Pointer mask_HR_reader = ScalarImageReaderType::New();
	     mask_HR_reader->SetFileName(maskImage_HR_n.c_str());
	     mask_HR_reader->Update();
	     ScalarImageType::Pointer maskImage_HR = mask_HR_reader->GetOutput();

	     // Read LR mask normally
	     const string maskImage_LR_n = cl.follow("NoFile", 1, "-mLR");
	     ScalarImageReaderType::Pointer mask_LR_reader = ScalarImageReaderType::New();
	     mask_LR_reader->SetFileName(maskImage_LR_n.c_str());
	     mask_LR_reader->Update();
	     ScalarImageType::Pointer maskImage_LR = mask_LR_reader->GetOutput();
*/
	     // Read Gradient Images
/*		const string file_gradImage_n = cl.follow("NoFile", 1, "-fG");
	     std::ifstream fileGImg(file_gradImage_n.c_str());

	     	int numOfGradImages = 0;
	     	fileGImg >> numOfGradImages;

	        typedef itk::Image<VectorType, ImageDim> VectorImageType;

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

	     	//Read LR Diffusion images
	        const string fileIn  = cl.follow("NoFile",1,"-iLR");

	     	ImageListType DWIList_LR;

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
	     	             DWIList_LR.push_back( myReader->GetOutput() ); //using push back to create a stack of diffusion images
	     	         }

	     	// Create Initial Tensor Estimate

	     		typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	     		typedef itk::Image<DiffusionTensorType, ImageDim> TensorImageType;
	    		UnweightedLeastSquaresTensorEstimation UnWeightedTensorEstimator;
	     		UnWeightedTensorEstimator.ReadDWIList(DWIList_HR);
	     		UnWeightedTensorEstimator.ReadMask(maskImage_HR);
	     		UnWeightedTensorEstimator.ReadBVal(1.0);
	     		UnWeightedTensorEstimator.ReadGradientList(gradientImageList);
	     		UnWeightedTensorEstimator.ReadB0Image(B0_HR);

	     		TensorImageType::Pointer tensorImage_init = UnWeightedTensorEstimator.Compute();

	     		typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;
	     		TensorWriterType::Pointer tensorWriter = TensorWriterType::New();
	     		tensorWriter->SetFileName("tensorImage_stupid.nii.gz");
	     		tensorWriter->SetInput(tensorImage_init);
	     		tensorWriter->Update();

	     		std::cout << "TensorComputeed " << std::endl;

			typedef itk::ImageFileReader<TensorImageType> TensorReaderType;
			TensorReaderType::Pointer TensorReader = TensorReaderType::New();
			TensorReader->SetFileName("tensorImage_stupid.nii.gz");
			TensorReader->Update();
			TensorImageType::Pointer tensorImage_init = TensorReader->GetOutput();			

	     	//Compute Sigma
	     		vnl_vector<RealType> Sigma;	     		
			ComputeSigma_LR computeSigma;

	     		computeSigma.ReadDWIList(DWIList_LR);
	     		computeSigma.ReadGradientList(gradientImageList);
	     		computeSigma.ReadMaskImage_HR(maskImage_HR);
	     		computeSigma.ReadB0Image_HR(B0_HR);
	     		computeSigma.ReadBVal(1.0);
	     		computeSigma.ReadTensorImage(tensorImage_init);
			computeSigma.ReadLRImage(maskImage_LR);
			computeSigma.ReadMapMatrix(MapHR2LR);
	     		Sigma = computeSigma.ComputeAttenuation();

*/


}



	     return 0;
}
