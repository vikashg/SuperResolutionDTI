/*
 * TestMappingDispField.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: vgupta
 */
#include <MapFilterLR2HRDispField.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageMaskSpatialObject.h"
#include "../GetPot/GetPot"
#include "ComposeImage.h"

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

	     const string fixedImage_n = cl.follow("NoFile", 1, "-f");
	     const string movingImage_n = cl.follow("NoFile",1, "-m");
	     const string dispField_n = cl.follow("NoFile", 1, "-d");
	     const string maskImage_n = cl.follow("NoFile",1, "-T1");

	     typedef float RealType;
	     typedef itk::Vector<double, 3> VectorType;
	     typedef itk::Image<VectorType, 3> DeformationFieldType;

	     typedef itk::ImageFileReader<DeformationFieldType> DisplacementFileReaderType;
	     DisplacementFileReaderType::Pointer deformationFieldReader = DisplacementFileReaderType::New();

	     deformationFieldReader->SetFileName(dispField_n.c_str());
	     deformationFieldReader->Update();

	     DeformationFieldType::Pointer defField = deformationFieldReader->GetOutput();

	     typedef itk::ImageMaskSpatialObject<3> MaskSpatiolObjectType;
	     typedef MaskSpatiolObjectType::ImageType MaskSpatialImageType;
	     typedef itk::ImageFileReader<MaskSpatialImageType> SpatialImageReaderType;
	     SpatialImageReaderType::Pointer readerSpatial = SpatialImageReaderType::New();
	     readerSpatial->SetFileName(maskImage_n.c_str());
	     readerSpatial->Update();

	     MaskSpatialImageType::Pointer maskImage = readerSpatial->GetOutput();

	     typedef itk::Image<RealType , 3> ImageType;
	     typedef itk::ImageFileReader<ImageType> ReaderType;
	     ReaderType::Pointer movingImageReader = ReaderType::New();

	     movingImageReader->SetFileName(movingImage_n.c_str());
	     movingImageReader->Update();
	     ImageType::Pointer T1 = movingImageReader->GetOutput();

	     ReaderType::Pointer fixedImageReader = ReaderType::New();
	     fixedImageReader->SetFileName(fixedImage_n.c_str());
	     fixedImageReader->Update();
	     ImageType::Pointer B0 = fixedImageReader->GetOutput();

	     MapFilterLR2HRDisp filter;
	     filter.ReadFixedImage(B0); //T1
	     filter.ReadMovingImage(T1); //B0
	     filter.ReadDeformationField(defField); // T1---> B0
	     filter.ReadMaskImage(maskImage);
	     filter.ComputeMapWithDefField();

	     std::cout << "Computed the Map " << std::endl;

	 	vnl_sparse_matrix<float> MapLR2HR, MapHR2LR;

	 	MapLR2HR = filter.GetLR2HRMatrix();
	 	MapHR2LR = filter.GetHR2LRMatrix();


		ofstream ofs("MapLR2HR.dat", ios::binary);
		ofs.write((char *)&MapLR2HR, sizeof(MapLR2HR));

		vnl_sparse_matrix<float> map1;

	 	ifstream ifs("MapLR2HR.dat", ios::binary);
		ifs.read((char *)&map1, sizeof(map1));


		ComposeImageFilter composeFilter;
	 	composeFilter.GetHRImage(T1);
	 	composeFilter.GetLRImage(B0);
	 	composeFilter.ReadMatrix(MapLR2HR);

	 	ImageType::Pointer tempImage = composeFilter.ComposeIt();
	 	typedef itk::ImageFileWriter<ImageType> WriterType;
	 	WriterType::Pointer writer = WriterType::New();
	 	writer->SetFileName("tempImageLR.nii.gz");
	 	writer->SetInput(tempImage);
	 	writer->Update();


		ComposeImageFilter composeFilter2;
		composeFilter2.GetHRImage(B0);
		composeFilter2.GetLRImage(T1);
		composeFilter2.ReadMatrix(MapHR2LR);
		
		ImageType::Pointer tempImageHR = composeFilter2.ComposeIt();
		
		WriterType::Pointer writer2 = WriterType::New();
		writer2->SetFileName("tempImageHR.nii.gz");
		writer2->SetInput(tempImageHR);
		writer2->Update();

	 	std::cout << T1->GetLargestPossibleRegion().GetSize() << std::endl;
	 	std::cout << B0->GetLargestPossibleRegion().GetSize() << std::endl;



	     return 0;
}
