#ifndef INC_WEIGHTEDLEASTSQUARES_CWLS_H_
#define INC_WEIGHTEDLEASTSQUARES_CWLS_H_

#include "TensorUtilites.h"
#include "itkDiffusionTensor3D.h"
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkMatrix.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_inverse.h"
#include "vnl/vnl_sparse_matrix.h"
#include "vnl/vnl_copy.h"
#include "itkSubtractImageFilter.h"
#include "itkImageFileWriter.h"

class WeightedLeastSquares
{
	typedef float RealType;
	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;


	typedef itk::Image<RealType, 3> ScalarImageType;
	typedef std::vector<ScalarImageType::Pointer> ImageListType;
	typedef itk::SubtractImageFilter<ScalarImageType> SubtractImageFilterType;
	
	typedef itk::Vector<double, 6> VectorialTensorType;
	typedef itk::Image<VectorialTensorType, 3> VectorialImageTensorType;	

        typedef itk::Vector<double, 3> VectorType;
        typedef itk::Image<VectorType, 3> VectorImageType;
        typedef std::vector<VectorImageType::Pointer> VectorImageListType;
	typedef vnl_vector<double> VnlVectorType;
	typedef vnl_matrix<RealType> MatrixType;

	typedef itk::ImageRegionIterator<ScalarImageType> ScalarImageIterator;
	typedef itk::ImageRegionIterator<VectorImageType> VectorImageIterator;
	typedef itk::ImageRegionIterator<TensorImageType> TensorImageIterator;
	typedef itk::ImageRegionIterator<VectorialImageTensorType> VecTensorImageIterator;
	typedef vnl_sparse_matrix<RealType> SparseMatrixType;

public:
	void ReadGradientList(VectorImageListType gradList);
	void ReadDWIListLR(ImageListType DWIList);
	void ReadDWIListHR(ImageListType DWIListHR);
	void ReadTensorImage(TensorImageType::Pointer tensorImage);

	void ReadHRMask(ScalarImageType::Pointer maskImage);
	void ReadLRMask(ScalarImageType::Pointer maskImage);

	void ReadBVal(RealType bVal);
	void ReadB0ImageHR(ScalarImageType::Pointer B0Image);
	void ReadB0ImageLR(ScalarImageType::Pointer B0Image);

	void ReadMapMatrixLR2HR(SparseMatrixType map);
	void ReadMapMatrixHR2LR(SparseMatrixType map);

	vnl_matrix<double> ComputeJacobian(DiffusionTensorType D);
	VnlVectorType ComputeWeightMatrixRow(VectorType G);

	ImageListType ComputePredictedImage(TensorImageType::Pointer tensorImage);
	
	ImageListType ComputeDifferenceImage( ImageListType PredImageList);
	VectorialImageTensorType::Pointer ComputeDelSim(TensorImageType::Pointer tensorImage);

	VectorialImageTensorType::Pointer ConvertDT2Vector(TensorImageType::Pointer tensorImage);	
	RealType NLSEnergy(TensorImageType::Pointer tensorImage);
	VectorialImageTensorType::Pointer MakeGammaImage(TensorImageType::Pointer tensorImage);
	TensorImageType::Pointer MakeTensorImage(VectorialImageTensorType::Pointer vecImage);	

	TensorImageType::Pointer ConvertGammaVector2DT(VectorialImageTensorType::Pointer vecTensorImage);
	
	void UpdateTerms();

private:
	void TensorEstimate();
	ImageListType m_DWIListLR, m_DWIListHR;
	VectorImageListType m_GradList;
	ScalarImageType::Pointer m_HRmask, m_LRmask;
	RealType m_BVal;
	TensorImageType::Pointer m_tensorImage_init;
	ScalarImageType::Pointer m_B0Image_LR, m_B0Image_HR;
 	SparseMatrixType m_MapHR2LR, m_MapLR2HR;	

};
#endif
