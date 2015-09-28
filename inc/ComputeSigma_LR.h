/* 
* ComputeSigma_LR.h

*/

#ifndef INC_COMPUTESIGMA_LR_H_
#define INC_COMPUTESIGMA_LR_H_

#include "itkBinaryThresholdImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_matrix.h"
#include "itkDiffusionTensor3D.h"
#include "itkImage.h"
#include "vnl/vnl_diag_matrix.h"
#include "itkImageRegionIterator.h"
#include "itkMaskImageFilter.h"
#include "itkNeighborhoodIterator.h"
#include "TensorUtilites.h"
#include "itkNeighborhoodIterator.h"
#include "itkAddImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkImageFileWriter.h"
#include "iostream"
#include "vnl/vnl_sparse_matrix.h"
#include "string"
#include "ComposeImage.h"
#include "vnl/vnl_copy.h"

class ComputeSigma_LR{
	   typedef float RealType;
	   typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
           typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

	   typedef itk::Image<RealType, 3> ScalarImageType;
	   typedef std::vector<ScalarImageType::Pointer> ImageListType;

	    typedef itk::ImageFileWriter<ScalarImageType>  WriterType;

	    typedef itk::Vector<double, 3> VectorType;
	    typedef itk::Image<VectorType, 3> VectorImageType;
	    typedef std::vector<VectorImageType::Pointer> VectorImageListType;
	    typedef vnl_matrix<RealType> MatrixType;

	    typedef itk::DivideImageFilter<ScalarImageType, ScalarImageType, ScalarImageType> DivideByImageFilterType;
		
	  typedef itk::MaskImageFilter<ScalarImageType, ScalarImageType> MaskImageFilterType;
	   typedef itk::SubtractImageFilter<ScalarImageType, ScalarImageType, ScalarImageType> SubtractImageFilterType;
	   	typedef itk::StatisticsImageFilter<ScalarImageType> StatisticsImageFilterType;

		typedef itk::ImageRegionIterator<ScalarImageType> ScalarImageIterator;
		typedef itk::ImageRegionIterator<VectorImageType> VectorImageIterator;
		typedef itk::ImageRegionIterator<TensorImageType> TensorImageIterator;
		typedef itk::NeighborhoodIterator<TensorImageType> TensorNeighIterator;
		typedef itk::NeighborhoodIterator<ScalarImageType> ScalarNeighIterator;
		typedef itk::ThresholdImageFilter<ScalarImageType> ThresholdImageFilterType;
		typedef itk::BinaryThresholdImageFilter<ScalarImageType, ScalarImageType> BinaryThresholdImageFilterType;
public:
		void ReadDWIList(ImageListType list);
		void ReadLRImage(ScalarImageType::Pointer LRImage);
		void ReadGradientList(VectorImageListType veclist);
		void ReadMaskImage_HR(ScalarImageType::Pointer maskImage);
		void ReadBVal(RealType bVal);
		void ReadB0Image_HR(ScalarImageType::Pointer B0Image);
		void ReadB0Image_LR(ScalarImageType::Pointer B0Image);
		void ReadMapMatrix(vnl_sparse_matrix<float> map);
		void ReadTensorImage(TensorImageType::Pointer tensorImage);
		vnl_vector<RealType> ComputeAttenuation();
		vnl_vector<RealType> ComputeAttenuation_Frac();


private:

		ImageListType m_DWIList;
		VectorImageListType m_GradList;
		ScalarImageType::Pointer m_maskImage_HR;
		TensorImageType::Pointer m_dt;
		RealType m_BVal;
		ScalarImageType::Pointer m_B0ImageHR, m_B0ImageLR, m_LRImage;
		
		vnl_sparse_matrix<float> m_MapHR2LR;

};

#endif /* INC_COMPUTESIGMA_LR_H_ */
