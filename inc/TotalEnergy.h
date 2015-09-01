/*
 * TotalEnergy.h
 *
 *  Created on: Aug 11, 2015
 *      Author: vgupta
 */

#ifndef INC_TOTALENERGY_H_
#define INC_TOTALENERGY_H_

#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_matrix.h"
#include "itkDiffusionTensor3D.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "vnl/vnl_diag_matrix.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "TensorUtilites.h"
#include "itkNeighborhoodIterator.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkDivideImageFilter.h"
#include "JointTensorEstimation.h"

class TotalEnergy{
	typedef float RealType;
	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

	typedef itk::Image<RealType, 3> ScalarImageType;
	typedef std::vector<ScalarImageType::Pointer> ImageListType;

    typedef itk::Vector<RealType, 3> VectorType;
    typedef itk::Image<VectorType, 3> VectorImageType;
    typedef std::vector<VectorImageType::Pointer> VectorImageListType;
    typedef vnl_matrix<RealType> MatrixType;

    typedef itk::MultiplyImageFilter<TensorImageType, TensorImageType, TensorImageType> MultiplyByScalarTensorImageFilterType;
    typedef itk::AddImageFilter<TensorImageType, TensorImageType, TensorImageType> AddTensorImageFilterType;

	typedef itk::ImageRegionIterator<ScalarImageType> ScalarImageIterator;
	typedef itk::ImageRegionIterator<VectorImageType> VectorImageIterator;
	typedef itk::ImageRegionIterator<TensorImageType> TensorImageIterator;
	typedef itk::NeighborhoodIterator<TensorImageType> TensorNeighIterator;
	typedef itk::NeighborhoodIterator<ScalarImageType> ScalarNeighIterator;

	typedef itk::ImageFileWriter<ScalarImageType> ScalarWriterType;
	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;

	typedef itk::DivideImageFilter<ScalarImageType, ScalarImageType, ScalarImageType> DivideImageFilterType;

public:
	RealType GaussianNoise(TensorImageType::Pointer logTensorImage);
	RealType RegularizationEnergy(TensorImageType::Pointer logTensorImage);
	void ReadImageList(ImageListType DWIList);
	void ReadGradientImageList(VectorImageListType GradientImageList);
	void ReadB0Image(ScalarImageType::Pointer image);
	void ReadMaskImage(ScalarImageType::Pointer maskImage);
	void ReadBValue(RealType bval);
	void ReadKappa(RealType kappa);
	void ReadSigma(vnl_vector<RealType> sigma);
	RealType GaussianNoise_Frac(TensorImageType::Pointer logTensorImage);

private:
	VectorImageListType m_GradList;
	ImageListType m_DWIList;
	ScalarImageType::Pointer m_MaskImage, m_B0Image;
	RealType m_BVal, m_Kappa;
	vnl_vector<RealType> m_Sigma;



};



#endif /* INC_TOTALENERGY_H_ */
