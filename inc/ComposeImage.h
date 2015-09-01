/*
 * ComposreImage.h
 *
 *  Created on: Jul 4, 2015
 *      Author: vgupta
 */

#ifndef INC_COMPOSEIMAGE_H_
#define INC_COMPOSEIMAGE_H_


#include "itkImage.h"
#include "itkPointSet.h"
#include "itkPoint.h"
#include <iostream>
#include <vnl/vnl_sparse_matrix.h>
#include <vcl_vector.h>
#include "itkTransform.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"

class ComposeImageFilter
{
	typedef itk::Image<float, 3> ImageType;
	typedef vnl_sparse_matrix<float> SparseMatrixType;
    typedef vnl_sparse_matrix_pair<float> pair_t;

public:
    ImageType::Pointer GetLRImage(ImageType::Pointer imageLR);
    ImageType::Pointer GetHRImage(ImageType::Pointer imageHR);
    void ReadMatrix(SparseMatrixType matrix);
    ImageType::Pointer ComposeIt();
 
private:
    ImageType::Pointer m_imageLR, m_imageHR;
    ImageType::IndexType ComputeImageIndex(long int N, ImageType::Pointer image);
    SparseMatrixType m_matrix;
    
};



#endif /* INC_COMPOSREIMAGE_H_ */
