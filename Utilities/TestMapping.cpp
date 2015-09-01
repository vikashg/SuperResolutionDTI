/*
 * TestMapping.cxx
 *
 *  Created on: Jul 2, 2015
 *      Author: vgupta
 */


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
#include "itkDiffusionTensor3D.h"
#include "itkVector.h"

using namespace std;
int main(int argc, char* *argv)
{


	     GetPot   cl(argc, argv);
	     if( cl.size() == 1 || cl.search(2, "--help", "-h") )
	      {
	          std::cout << "Not enough arguments" << std::endl;
	          return -1;
	      }

	     const string LR_file_n = cl.follow("NoFile", 1 , "-LR");
	     const string HR_file_n = cl.follow("NoFile", 1 , "-HR");
	     const string output_file_n = cl.follow("Nofile.nii.gz", 1, "-o");
	     const string trans_n   = cl.follow("NoFile", 1, "-t");
	     const string disp_n = cl.follow("NoFile", 1, "-d");

	     typedef itk::Image<float, 3> ImageType;
	     typedef itk::ImageFileReader<ImageType> ReaderType;
	     typedef itk::ImageFileWriter<ImageType> WriterType;

	     typedef itk::TransformFileReader TransformFileReaderType;
	     typedef TransformFileReaderType::TransformListType TransformListType;
	     typedef itk::TransformBase TransformBaseType;
	     typedef itk::AffineTransform<double, 3> AffineTransformType;

	     typedef itk::DiffusionTensor3D<float> DiffusionTensorType;
	     typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

	     typedef itk::Vector<float, 3> VectorType;
	     typedef itk::Image<VectorType, 3> DispImageType;
	 
	ReaderType::Pointer readerLR = ReaderType::New();
	readerLR->SetFileName(LR_file_n);
	readerLR->Update();

	ImageType::Pointer imageLR = readerLR->GetOutput();

	ReaderType::Pointer readerHR = ReaderType::New();
	readerHR->SetFileName(HR_file_n);
	readerHR->Update();
	ImageType::Pointer imageHR = readerHR->GetOutput();


	typedef itk::ImageFileReader<DispImageType> DispImageFilterType;
	DispImageFilterType::Pointer dispImageReader = DispImageFilterType::New();
	dispImageReader->SetFileName(disp_n);
	dispImageReader->Update();
	DispImageType::Pointer dispField = dispImageReader->GetOutput();


	std::cout << "Read Images"  << std::endl;
	//Reading transforms
	TransformFileReaderType::Pointer readerTransform = TransformFileReaderType::New();
	readerTransform->SetFileName(trans_n);
	readerTransform -> Update();
	TransformListType *list = readerTransform->GetTransformList();
	TransformBaseType * transform = list->front().GetPointer();
	TransformBaseType::ParametersType parameters = transform->GetParameters();
	AffineTransformType::Pointer transform_fwd = AffineTransformType::New();
	transform_fwd->SetParameters(parameters);

	MapFilterLR2HR1 filter;
	filter.ReadHRImage(imageHR);
	filter.ReadLRImage(imageLR);
	filter.ReadAffineTransform(transform_fwd);	
	filter.ReadDeformationField(dispField);
	filter.ComputeMapWithDefField();
		

	vnl_sparse_matrix<float> MapLR2HR, MapHR2LR;

	MapLR2HR = filter.GetLR2HRMatrix();
//	MapHR2LR = filter.GetHR2LRMatrix();

	ComposeImageFilter composeFilter;
	composeFilter.GetHRImage(imageHR);
	composeFilter.GetLRImage(imageLR);
	composeFilter.ReadMatrix(MapLR2HR);
	ImageType::Pointer image = composeFilter.ComposeIt();

	std::cout << "composing done" << std::endl;

	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(output_file_n);
	writer->SetInput(image);
	writer->Update();

	std::cout << "Map generated" << std::endl;

	return 0;
}

