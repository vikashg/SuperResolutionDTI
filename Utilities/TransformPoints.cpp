/*
 * TransformPoints.cpp
 *
 *  Created on: Aug 29, 2015
 *      Author: vgupta
 */

#define TOTAL_PTS 10000000


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkDisplacementFieldTransform.h"
#include "itkImageFileWriter.h"
#include "../GetPot/GetPot"
#include "itkPoint.h"
#include "itkImageRegionIterator.h"
#include "itkImageMaskSpatialObject.h"
#include "CopyImage.h"
#include "itkImageFileWriter.h"
#include <fstream>
#include <iostream>


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

    const string defField_n = cl.follow("NoFile",1 , "-d");
    const string fixedImage_n = cl.follow("NoFile", 1, "-f");
    const string movingImage_n = cl.follow("NoFile", 1, "-m");


    typedef double RealType;

    typedef itk::Image<RealType, 3> ImageType;
    typedef itk::ImageFileReader<ImageType> ImageReaderType;

    ImageReaderType::Pointer fixedImageReader = ImageReaderType::New();
    ImageReaderType::Pointer movingImageReader = ImageReaderType::New();


    typedef itk::ImageMaskSpatialObject<3> MaskSpatialObjectType;
    typedef MaskSpatialObjectType::ImageType MaskSpatialImageType;

    typedef itk::ImageFileReader<MaskSpatialImageType> MaskSpatialImageReader;



    typedef itk::Vector<RealType,3> VectorType;
    typedef itk::Image<VectorType , 3> DisplacementFieldImageType;
    typedef itk::ImageFileReader<DisplacementFieldImageType> DisplacementFieldReaderType;

    DisplacementFieldReaderType::Pointer dispReader = DisplacementFieldReaderType::New();
    dispReader->SetFileName(defField_n.c_str());
    dispReader->Update();

    DisplacementFieldImageType::Pointer defField = dispReader->GetOutput();

    typedef itk::DisplacementFieldTransform<RealType,3> DeformationFieldTransformType;
    DeformationFieldTransformType::Pointer defFieldTransform = DeformationFieldTransformType::New();
    defFieldTransform->SetDisplacementField(defField);


    MaskSpatialImageReader::Pointer maskSpatialImageReader = MaskSpatialImageReader::New();
    maskSpatialImageReader->SetFileName(fixedImage_n.c_str());
    maskSpatialImageReader->Update();

    MaskSpatialImageType::Pointer fixedImage = maskSpatialImageReader->GetOutput();

    movingImageReader->SetFileName(movingImage_n.c_str());
    movingImageReader->Update();

    ImageType::Pointer movingImage = movingImageReader->GetOutput();

    typedef itk::Point<RealType, 3> PointType;
//    PointType P1;
//
//    P1=fixedImage->GetOrigin();
//    std::cout << P1 << std::endl;
//    std::cout << defFieldTransform->TransformPoint(P1) << std::endl;



    MaskSpatialObjectType::Pointer maskSO = MaskSpatialObjectType::New();
    maskSO->SetImage(fixedImage);

    ImageType::PointType m_Origin_LR, m_Origin_HR;
    m_Origin_HR = fixedImage->GetOrigin();
    m_Origin_LR = movingImage->GetOrigin();

    ImageType::SizeType sizeLR, sizeHR;
    sizeLR = movingImage->GetLargestPossibleRegion().GetSize();
    sizeHR = fixedImage->GetLargestPossibleRegion().GetSize();

    ImageType::PointType m_Final_HR, m_Final_LR;
    MaskSpatialImageType::IndexType IndexFinal;

    for (int i=0; i < 3; i++)
    {
    	IndexFinal[i] = sizeHR[i] -1;
    }

    fixedImage->TransformIndexToPhysicalPoint(IndexFinal, m_Final_HR);

    ImageType::Pointer outputImage = ImageType::New();
//
//    outputImage->SetOrigin(movingImage->GetOrigin());
//    outputImage->SetSpacing(movingImage->GetSpacing());
//    outputImage->SetDirection(movingImage->GetDirection());
//    outputImage->SetRegions(movingImage->GetLargestPossibleRegion());
//    outputImage->Allocate();
//    outputImage->FillBuffer(0.0);
//    outputImage = movingImage;


    std::cout << "Output Image" << std::endl;


    int num =0;

        while (num < TOTAL_PTS)
        {
        	PointType P_LR, P_HR;

        	for (int j=0; j < 3; j++)
        	{
       		 double r =  ((double) rand() /(RAND_MAX)) ;
     		 double temp_add = r*(m_Final_HR[j] - m_Origin_HR[j]);
     		     P_HR[j] = m_Origin_HR[j] + temp_add;

        	}

        	if (maskSO->IsInside(P_HR) == 1)
        	{
        		P_LR = defFieldTransform->TransformPoint(P_HR);
        	}

        	ImageType::IndexType a_index;
        	movingImage->TransformPhysicalPointToIndex(P_LR, a_index);

        	movingImage->SetPixel(a_index, 5000);

        	num++;


        }

    typedef itk::ImageFileWriter<ImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();

    writer->SetFileName("Output_Image.nii.gz");
    writer->SetInput(movingImage);
    writer->Update();


    return 0;
}

