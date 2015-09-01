#ifndef ITKMAPLR2HR_H
#define ITKMAPLR2HR_H

#include "itkImage.h"
#include "itkPointSet.h"
#include "itkPoint.h"
#include "itkImageToImageFilter.h"
#include <iostream>
#include <fstream>
#include <vnl/vnl_sparse_matrix.h>
#include <vcl_vector.h>
#include "itkTransform.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"

namespace itk
{
template<class TImage>
class MapLR2HRFilter:public ImageToImageFilter<TImage, TImage>
{
public:
    typedef MapLR2HRFilter Self;
    typedef ImageToImageFilter<TImage, TImage> Superclass;
    typedef SmartPointer<Self> Pointer;

       typedef itk::PointSet<double, 3> PointSetType;
       typedef PointSetType::PointType PointType;
       typedef PointSetType::PointsContainerPointer PointsContainerPointer;
       typedef itk::TransformFileReader TransformFileReaderType;
       typedef TransformFileReaderType::TransformListType TransformListType;
       typedef itk::TransformBase TransformBaseType;
       typedef itk::AffineTransform<double, 3> AffineTransformType;

       typedef vnl_sparse_matrix<float> SpMatType;

       itkNewMacro( Self );
       itkTypeMacro(MapLR2HRFilter, ImageToImageFilter);

       void SetLRImage(const TImage* imageLR);
       void SetHRImage(const TImage* imageHR);
       void SetTransform( AffineTransformType::Pointer  transform);



protected:


       MapLR2HRFilter();
         ~MapLR2HRFilter(){};

       typename TImage::ConstPointer GetLRImage();
       typename TImage::ConstPointer GetHRImage();
       virtual void GenerateInputRequestedRegion();
       virtual void GenerateData();
       unsigned long int ComputeMatrixIndex(typename TImage::IndexType, typename TImage::ConstPointer);


private:


       typename TImage::Pointer m_ImageLR, m_ImageHR;
       AffineTransformType::Pointer m_Transform;
       vnl_sparse_matrix<float> m_SpVnl_Row_normalize, m_SpVnl_Col_normalize;

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMapLR2HR.hxx"
#endif

#endif
