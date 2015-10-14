/*
 * JointTensorEstimation.h
 *
 *  Created on: Jul 28, 2015
 *      Author: vgupta
 */

#ifndef INC_JOINTTENSORESTIMATION_H_
#define INC_JOINTTENSORESTIMATION_H_

#include "itkSquareImageFilter.h"
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
#include "TotalEnergy.h"
#include "vnl/vnl_sparse_matrix.h"
#include "ComposeImage.h"
#include "itkSubtractImageFilter.h"
#include "vnl/vnl_copy.h"

class JointTensorEstimation{
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
    typedef itk::DivideImageFilter<ScalarImageType, ScalarImageType, ScalarImageType> DivideByImageFilterType;
    typedef itk::SquareImageFilter<ScalarImageType, ScalarImageType> SquareImageFilterType;
    typedef itk::SubtractImageFilter<ScalarImageType, ScalarImageType, ScalarImageType> SubtractImageFilterType;
	typedef itk::ImageRegionIterator<ScalarImageType> ScalarImageIterator;
	typedef itk::ImageRegionIterator<VectorImageType> VectorImageIterator;
	typedef itk::ImageRegionIterator<TensorImageType> TensorImageIterator;
	typedef itk::NeighborhoodIterator<TensorImageType> TensorNeighIterator;
	typedef itk::NeighborhoodIterator<ScalarImageType> ScalarNeighIterator;
	
	typedef itk::ImageFileWriter<ScalarImageType> ScalarWriterType;
	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;

	typedef vnl_sparse_matrix<RealType> SparseMatrixType;



public:
	void ReadDWIListHR(ImageListType list);
	void ReadDWIListLR(ImageListType list);

	void ReadGradientList(VectorImageListType veclist);
	void ReadLambda(RealType lambda);
	void ReadSigma(vnl_vector<RealType> sigma);
	void ReadBVal(RealType Bval);
	void ReadInitialTensorImage(TensorImageType::Pointer tensorImage);
	TensorImageType::Pointer ComputeDelSim();
	TensorImageType::Pointer UpdateTerms1();
	void ReadKappa(RealType kappa);
	void ReadB0ImageHR(ScalarImageType::Pointer B0Image);
	void ReadB0ImageLR(ScalarImageType::Pointer B0Image);


	void ReadStepSize(RealType m_StepSize);
	void ReadNumOfIterations(int m_Iterations);
	void ReadB0Thres(RealType thres);

	void ReadMapMatrixLR2HR(SparseMatrixType Map);
	void ReadMapMatrixHR2LR(SparseMatrixType Map);

	void ReadHRMask(ScalarImageType::Pointer image);
	void ReadLRMask(ScalarImageType::Pointer image);	

	ScalarImageType::Pointer ComputePsiImage(ScalarImageType::Pointer image); // tested OK
	ScalarImageType::Pointer GradientLogMagTensorImage(TensorImageType::Pointer logTensorImage); //checked tested OK
	TensorImageType::Pointer ComputeLaplaceTensor(TensorImageType::Pointer logTensorImage); //checked tested OK

	TensorImageType::Pointer ComputeFirstTermDelReg(ScalarImageType::Pointer PsiImage, TensorImageType::Pointer LaplaceTensorImage);
	TensorImageType::Pointer ComputeSecondTermDelReg(ScalarImageType::Pointer PsiImage, TensorImageType::Pointer LogTensorImage);
	TensorImageType::Pointer ComputeDelReg(TensorImageType::Pointer FirstTerm, TensorImageType::Pointer SecTerm);

	TensorImageType::Pointer ComputeDelSim_Frac(TensorImageType::Pointer LogtensorImage);
	TensorImageType::Pointer ComputeDelSim_nonFrac(TensorImageType::Pointer LogtensorImage);

	ImageListType ComputeDifferenceImages_Frac(TensorImageType::Pointer logTensorImage);  //Checked
	ImageListType ComputeDifferenceImages(TensorImageType::Pointer logTensorImage);       //Checked

	ImageListType ComputeAttenuation(TensorImageType::Pointer logTensorImage);
	ImageListType ComputeAttenuation_Frac(TensorImageType::Pointer logTensorImage);

	TensorImageType::Pointer ComputeDelSim_Frac_DispField(TensorImageType::Pointer LogTensorImage); 
	TensorImageType::Pointer ComputeDelSim_DispField(TensorImageType::Pointer LogTensorImage); 


private:
	void GaussianNoise();
	ImageListType m_DWIListHR, m_DWIListLR;
	VectorImageListType m_GradList;
	ScalarImageType::Pointer m_maskImage;
	RealType m_Lambda;
	vnl_vector<RealType> m_Sigma;
	TensorImageType::Pointer m_dt_init;
	RealType m_bval, m_kappa, m_Step_size;
	int m_Iterations;
	ScalarImageType::Pointer m_B0Image_HR, m_B0Image_LR;
	RealType m_thres;
	SparseMatrixType m_MapHR2LR, m_MapLR2HR;
	ScalarImageType::Pointer m_LRmask, m_HRmask;
};



#endif /* INC_JOINTTENSORESTIMATION_H_ */
