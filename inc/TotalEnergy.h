/*
 * TotalEnergy.h
 *
 *  Created on: Aug 11, 2015
 *      Author: vgupta
 */

#ifndef INC_TOTALENERGY_H_
#define INC_TOTALENERGY_H_

#include "itkThresholdImageFilter.h"
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
#include "ComposeImage.h"


class TotalEnergy{
	typedef float RealType;
	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

	typedef itk::Image<RealType, 3> ScalarImageType;
	typedef std::vector<ScalarImageType::Pointer> ImageListType;

    typedef itk::Vector<double, 3> VectorType;
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

	 typedef vnl_sparse_matrix<RealType> SparseMatrixType;

public:
	RealType GaussianNoise(TensorImageType::Pointer logTensorImage);
	RealType RegularizationEnergy(TensorImageType::Pointer logTensorImage);
	void ReadObsImageList(ImageListType DWIList);
	void ReadPredImageList(ImageListType PredList);
	void ReadGradientImageList(VectorImageListType GradientImageList);
	void ReadB0Image(ScalarImageType::Pointer image);
	void ReadLRMaskImage(ScalarImageType::Pointer maskLRImage);
	void ReadHRMaskImage(ScalarImageType::Pointer maskHRImage);
	void ReadBValue(RealType bval);
	void ReadKappa(RealType kappa);
	void ReadMapMatrixHR2LR(SparseMatrixType Map);
	void ReadSigma(vnl_vector<RealType> sigma);
	
	void SetFracFlag(bool flag);	

	RealType GaussianNoise_Frac(TensorImageType::Pointer logTensorImage);
	RealType GaussianNoise_woSR(TensorImageType::Pointer logTensorImage);
	
	RealType GaussianNoise_SR(TensorImageType::Pointer logTensorImage);
private:
	VectorImageListType m_GradList;
	ImageListType m_DWIList, m_predList;
	ScalarImageType::Pointer m_LRMaskImage, m_HRMaskImage,  m_B0Image;
	RealType m_BVal, m_Kappa;
	vnl_vector<RealType> m_Sigma;
	bool m_FracFlag;
	
	SparseMatrixType m_MapHR2LR;

};



#endif /* INC_TOTALENERGY_H_ */
