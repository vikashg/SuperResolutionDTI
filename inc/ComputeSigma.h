/*
 * ComputeSigma.h
 *
 *  Created on: Aug 2, 2015
 *      Author: vgupta
 */

#ifndef INC_COMPUTESIGMA_H_
#define INC_COMPUTESIGMA_H_


#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_matrix.h"
#include "itkDiffusionTensor3D.h"
#include "itkImage.h"
#include "vnl/vnl_diag_matrix.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "TensorUtilites.h"
#include "itkNeighborhoodIterator.h"
#include "itkAddImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkImageFileWriter.h"
#include "iostream"
#include "string"


class ComputeSigma{
		typedef float RealType;
		typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
		typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

		typedef itk::Image<RealType, 3> ScalarImageType;
		typedef std::vector<ScalarImageType::Pointer> ImageListType;

		typedef itk::ImageFileWriter<ScalarImageType>  WriterType;

	    typedef itk::Vector<RealType, 3> VectorType;
	    typedef itk::Image<VectorType, 3> VectorImageType;
	    typedef std::vector<VectorImageType::Pointer> VectorImageListType;
	    typedef vnl_matrix<RealType> MatrixType;

	    typedef itk::DivideImageFilter<ScalarImageType, ScalarImageType, ScalarImageType> DivideByImageFilterType;

	   	typedef itk::SubtractImageFilter<ScalarImageType, ScalarImageType, ScalarImageType> SubtractImageFilterType;
	   	typedef itk::StatisticsImageFilter<ScalarImageType> StatisticsImageFilterType;

		typedef itk::ImageRegionIterator<ScalarImageType> ScalarImageIterator;
		typedef itk::ImageRegionIterator<VectorImageType> VectorImageIterator;
		typedef itk::ImageRegionIterator<TensorImageType> TensorImageIterator;
		typedef itk::NeighborhoodIterator<TensorImageType> TensorNeighIterator;
		typedef itk::NeighborhoodIterator<ScalarImageType> ScalarNeighIterator;


public:
		void ReadDWIList(ImageListType list);
		void ReadGradientList(VectorImageListType veclist);
		void ReadMaskImage(ScalarImageType::Pointer maskImage);
		void ReadBVal(RealType bVal);
		void ReadB0Image(ScalarImageType::Pointer B0Image);
		void ReadTensorImage(TensorImageType::Pointer tensorImage);
		vnl_vector<RealType> ComputeAttenuation();
		vnl_vector<RealType> ComputeAttenuation_Frac();


private:

		ImageListType m_DWIList;
		VectorImageListType m_GradList;
		ScalarImageType::Pointer m_maskImage;
		TensorImageType::Pointer m_dt;
		RealType m_BVal;
		ScalarImageType::Pointer m_B0Image;
};



#endif /* INC_COMPUTESIGMA_H_ */
