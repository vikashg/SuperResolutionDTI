/*
 * StepWiseTransformGradients.cpp
 *
 *  Created on: Aug 27, 2015
 *      Author: vgupta
 */

#include "itkImage.h"
#include "../GetPot/GetPot"
#include "itkTransformFileReader.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkAffineTransform.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

using namespace std;
int main(int argc, char *argv[])
{
    GetPot cl (argc, const_cast<char**>(argv));
    if( cl.size() == 1 || cl.search (2,"--help","-h") )
	    {
	        std::cout << "Not Enough Arguments" << std::endl;
	        std::cout << "Generate the Gradient Table" << std::endl;
	        std::cout << "Usage:  return -1" << std::endl;
	    }

    // One idea is to apply the rotation matrix
    const string Linear_trans_n = cl.follow("NoFile",1,"-t");
    const string ref_Image_n = cl.follow("NoFile",1, "-r");
    const string out_n = cl.follow("NoFile",1,"-o");
    const string B0_n =cl.follow("NoFile",1,"-B0");
    const string T1_n =cl.follow("NoFile",1,"-T1");

    typedef itk::Image<float, 3> ImageType;
    typedef itk::ImageFileReader<ImageType> ImageReaderType;
    ImageReaderType::Pointer imageReaderB0 = ImageReaderType::New();
    imageReaderB0->SetFileName(B0_n);
    imageReaderB0->Update();

    std::cout << imageReaderB0->GetOutput()->GetDirection() << std::endl;


    ImageReaderType::Pointer imageReaderT1 = ImageReaderType::New();

    imageReaderT1->SetFileName(B0_n);
        imageReaderT1->Update();


        std::cout << imageReaderT1->GetOutput()->GetDirection() << std::endl;


//    typedef itk::TransformFileReader TransformFileReaderType;
//    typedef TransformFileReaderType::TransformListType TransformListType;
//    typedef itk::TransformBase TransformBaseType;
//    typedef itk::AffineTransform<double, 3> AffineTransformType;
//
//    typedef itk::Image<float, 3> ImageType;
//    typedef itk::ImageFileReader<ImageType> ImageReaderType;
//    ImageReaderType::Pointer imageReader = ImageReaderType::New();
//
//    imageReader->SetFileName(ref_Image_n);
//    imageReader->Update();
//    ImageType::Pointer refImage = imageReader->GetOutput();
//
//
//
//	TransformFileReaderType::Pointer readerTransform = TransformFileReaderType::New();
//	readerTransform->SetFileName(Linear_trans_n);
//	readerTransform -> Update();
//	TransformListType *list = readerTransform->GetTransformList();
//	TransformBaseType * transform = list->front().GetPointer();
//	TransformBaseType::ParametersType parameters = transform->GetParameters();
//	AffineTransformType::Pointer transform_fwd = AffineTransformType::New();
//	transform_fwd->SetParameters(parameters);
//
//	std::cout << transform_fwd->GetParameters() << std::endl;

//    typedef itk::Vector< float, 3 >          VectorPixelType;
//    typedef itk::Image< VectorPixelType, 3 > DisplacementFieldImageType;
//
//
//    typedef itk::TransformToDisplacementFieldFilter<DisplacementFieldImageType, double> DisplacementFieldGeneratorType;
//    DisplacementFieldGeneratorType::Pointer dispfieldGenerator = DisplacementFieldGeneratorType::New();
//
//    dispfieldGenerator->UseReferenceImageOn();
//    dispfieldGenerator->SetReferenceImage( refImage );
//    dispfieldGenerator->SetTransform( transform_fwd );
//    dispfieldGenerator->Update();
//    DisplacementFieldImageType::Pointer dispField = dispfieldGenerator->GetOutput();
//
//    typedef itk::ImageFileWriter<DisplacementFieldImageType> DispFieldWriterType;
//    DispFieldWriterType::Pointer writer = DispFieldWriterType::New();
//    writer->SetFileName(out_n);
//    writer->SetInput(dispField);
//    writer->Update();

    return 0;
}
