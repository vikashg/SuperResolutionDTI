#ifndef INC_TRANSFORMGRADIENTS_H_
#define INC_TRANSFORMGRADIENTS_H_

#include "itkImage.h"
#include "itkDeformationFieldGradientTensorImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "vnl/algo/vnl_svd.h"

class TransformGradients{
	typedef double RealType;
	typedef itk::Vector<RealType, 3> VectorType;
	typedef itk::Image<VectorType, 3> VectorImageType;
	typedef itk::Image<RealType, 3> ScalarImageType;	

	typedef std::vector<VectorType> GradientListType;
	typedef itk::Image<float, 3> ScalarFloatImageType;

	typedef itk::Matrix<double,3,3> MatrixType;
	typedef itk::Image<MatrixType, 3> MatrixImageType;	

	typedef itk::ImageRegionIterator<VectorImageType> VectorIterator;
	typedef itk::ImageRegionIterator<ScalarImageType> ScalarIterator;	
	typedef itk::ImageRegionIterator<MatrixImageType> MatrixImageIterator;
	typedef itk::ImageRegionIterator<ScalarFloatImageType> ScalarFloatIterator;

	typedef itk::DeformationFieldGradientTensorImageFilter<VectorImageType, double> JacobianFilterType;	

	typedef std::vector<VectorImageType::Pointer> VectorImageListType;	
		
public:
	void ReadMaskImage(ScalarFloatImageType::Pointer maskImage);
	void ReadDeformationField(VectorImageType::Pointer defField);
	void ReadGradients(GradientListType list);
	void ComputeGradients();
	VectorImageListType GetGradientImages();
	
private:
	VectorImageType::Pointer m_defField;
	ScalarFloatImageType::Pointer m_MaskImage;
	GradientListType m_GradList;
	VectorImageListType m_GradientImages;
};

#endif /* INC_TRANSFORMGRADIENTS_H_*/
