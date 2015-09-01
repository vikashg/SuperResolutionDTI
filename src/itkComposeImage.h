#ifndef ITKCOMPOSEIMAGE_H
#define ITKCOMPOSEIMAGE_H

#include "itkImage.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImageToImageFilter.h"
#include <vnl/vnl_sparse_matrix.h>
#include <cmath>
//#include <stdlib.h>
#include <cstdlib>

namespace itk
{

template<class TImage>
class ComposeImageFilter:public ImageToImageFilter<TImage, TImage>
{
public:
    typedef ComposeImageFilter Self;
    typedef ImageToImageFilter<TImage, TImage> Superclass;
    typedef SmartPointer<Self> Pointer;

    typedef vnl_sparse_matrix<float> SparseMatrixType;
    typedef vnl_sparse_matrix_pair<float> pair_t;

    itkNewMacro( Self )
    itkTypeMacro( ComposeImageFilter, ImageToImageFilter)

    void SetSourceImage(const TImage* imageSrc);
    void SetTargetImage(const TImage* imageTrg);
    void ReadMapMatrix(vnl_sparse_matrix<float> matrix);

    TImage* GetOutput();


protected:
    ComposeImageFilter();
    ~ComposeImageFilter(){};

    typename TImage::ConstPointer GetSourceImage();
    typename TImage::ConstPointer GetTargetImage();

    /*Write Output*/
    DataObject::Pointer MakeOutput(unsigned int idx);

    virtual void GenerateInputRequestedRegion();
    virtual void GenerateData();

    typename TImage::IndexType ComputeImageIndex( long int, typename TImage::ConstPointer);

private:

    SparseMatrixType m_matrix;



};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkComposeImage.txx"
#endif


#endif // ITKCOMPOSEIMAGE_H
