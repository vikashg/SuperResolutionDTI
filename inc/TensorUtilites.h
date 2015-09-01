/*
 * TensorUtilites.h
 *
 *  Created on: Jul 27, 2015
 *      Author: vgupta
 */

#ifndef INC_TENSORUTILITES_H_
#define INC_TENSORUTILITES_H_

#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_matrix.h"
#include "itkDiffusionTensor3D.h"
#include "itkImage.h"
#include "vnl/vnl_diag_matrix.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "CopyImage.h"

class TensorUtilities
{
	typedef float RealType;
	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;
	typedef itk::Image<RealType, 3> ScalarImageType;
	typedef itk::ImageRegionIterator<TensorImageType> TensorIterator;
	typedef itk::ImageRegionIterator<ScalarImageType> ScalarIterator;
	typedef itk::NeighborhoodIterator<TensorImageType> TensorNeighIterator;
	typedef vnl_matrix<RealType> MatrixType;
	typedef itk::Vector<RealType, 3 > VectorType;


public:
	DiffusionTensorType LogM(DiffusionTensorType D);
	DiffusionTensorType ExpM(DiffusionTensorType D);
	TensorImageType::Pointer ReplaceNaNsReverseEigenValue(TensorImageType::Pointer tensorImage, ScalarImageType::Pointer maskImage);
	MatrixType ConvertDT2Mat(DiffusionTensorType D);
	RealType ComputeTensorNorm(DiffusionTensorType D);

	DiffusionTensorType ConvertMat2DT(MatrixType D_mat);
	TensorImageType::Pointer LogTensorImageFilter(TensorImageType::Pointer tensorImage, ScalarImageType::Pointer maskImage);
	TensorImageType::Pointer ExpTensorImageFilter(TensorImageType::Pointer tensorImage, ScalarImageType::Pointer maskImage);
	DiffusionTensorType MatrixExpDirDerivative(DiffusionTensorType D, VectorType G, ScalarImageType::IndexType Idx);
};




#endif /* INC_TENSORUTILITES_H_ */
