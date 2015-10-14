/*
 * JointTensorEstimation.cxx
 *
 *  Created on: Jul 28, 2015
 *      Author: vgupta
 */


#include "math.h"
#include "JointTensorEstimation_woSR.h"
#include "itkImageFileWriter.h"
#include "math.h"


using namespace std;
void JointTensorEstimation::ReadDWIListLR(ImageListType list)
{
	m_DWIListLR = list;
}

void JointTensorEstimation::ReadDWIListHR(ImageListType list)
{
	m_DWIListHR = list;
}

void JointTensorEstimation::ReadLRMask(ScalarImageType::Pointer image)
{
	m_LRmask = image;
}

void JointTensorEstimation::ReadHRMask(ScalarImageType::Pointer image)
{
	m_HRmask = image;
}

void JointTensorEstimation::ReadGradientList(VectorImageListType vecList)
{
	m_GradList = vecList;
}

void JointTensorEstimation::ReadLambda(RealType lambda)
{
	m_Lambda = lambda;
}

void JointTensorEstimation::ReadKappa(RealType kappa)
{
	m_kappa =kappa;
}

void JointTensorEstimation::ReadBVal(RealType b_val)
{
	m_bval = b_val;
}

void JointTensorEstimation::ReadSigma(vnl_vector<RealType> sigma)
{
	m_Sigma = sigma;
}

void JointTensorEstimation::ReadInitialTensorImage(TensorImageType::Pointer tensorImage)
{
	m_dt_init = tensorImage;
}


void JointTensorEstimation::ReadStepSize(RealType step_size)
{
	m_Step_size =step_size;
}

void JointTensorEstimation::ReadNumOfIterations(int num)
{
	m_Iterations =num;
}

void JointTensorEstimation::ReadB0ImageHR(ScalarImageType::Pointer image)
{
	m_B0Image_HR = image;
}

void JointTensorEstimation::ReadB0ImageLR(ScalarImageType::Pointer image)
{
	m_B0Image_LR = image;
}

void JointTensorEstimation::ReadB0Thres(RealType thres)
{
		m_thres = thres;
}

void JointTensorEstimation::ReadMapMatrixLR2HR(SparseMatrixType map)
{
	m_MapLR2HR = map;
}

void JointTensorEstimation::ReadMapMatrixHR2LR(SparseMatrixType map)
{
	m_MapHR2LR = map;
}

JointTensorEstimation::ImageListType JointTensorEstimation::ComputeDifferenceImages_Frac(TensorImageType::Pointer logTensorImage)
{
       	int numOfImages = m_DWIListHR.size();
         TensorUtilities utilsTensor;
         CopyImage cpImage;
         TensorImageType::Pointer tensorImage = utilsTensor.ExpTensorImageFilter (logTensorImage, m_HRmask);

	ImageListType DiffImageList;
	
	for (int i =0; i < numOfImages; i++)
	{


           ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
                 ScalarImageIterator itB0(m_B0Image_HR, m_B0Image_HR->GetLargestPossibleRegion());
                 TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());

                ScalarImageType::Pointer diffImage_HR_i = ScalarImageType::New();
                 cpImage.CopyScalarImage(m_B0Image_HR, diffImage_HR_i);
                 ScalarImageIterator itDiff(diffImage_HR_i, diffImage_HR_i->GetLargestPossibleRegion());

                 ScalarImageType::Pointer attenImage_i = ScalarImageType::New();
                 cpImage.CopyScalarImage(m_B0Image_HR, attenImage_i);

                 for (itDiff.GoToBegin(), itHRMask.GoToBegin(), itB0.GoToBegin(), itTens.GoToBegin();
                       !itDiff.IsAtEnd(), !itHRMask.IsAtEnd(), !itB0.IsAtEnd(), !itTens.IsAtEnd(); 
			++itDiff, ++itHRMask, ++itB0, ++itTens)
                 {
                    if (itHRMask.Get() != 0)
                    {

                   RealType atten_i;
                   vnl_vector<double> g_i_temp= m_GradList[i]->GetPixel(itHRMask.GetIndex()).GetVnlVector();
                   vnl_vector<RealType> g_i; g_i.set_size(3);
                   vnl_copy(g_i_temp, g_i);

                 vnl_matrix<RealType> g_mat_i;
                 g_mat_i.set_size(3,1);
                 g_mat_i.set_column(0,g_i);

                 DiffusionTensorType D = itTens.Get();
                 MatrixType D_mat;
                 D_mat.set_size(3,3);
                 D_mat = utilsTensor.ConvertDT2Mat(D);
                  MatrixType temp; temp.set_size(1,1);
                 temp = g_mat_i.transpose()*D_mat*g_mat_i;

                 atten_i = exp(temp(0,0)*(-1)*m_bval);
	
		RealType temp1;
		temp1 = m_DWIListHR[i]->GetPixel(itHRMask.GetIndex())/itB0.Get() - atten_i;
		
		itDiff.Set(temp1);

                }
               }


		typedef itk::ImageFileWriter<ScalarImageType> ScalarImageWriterType;
		ScalarImageWriterType::Pointer writer = ScalarImageWriterType::New();

		   int num =i;
                 std::ostringstream num_con;
                 num_con << num;
                 std::string result  = num_con.str();

		std::string tempString = "Diff_" + result + ".nii.gz";
		
		writer->SetFileName(tempString);
		writer->SetInput(diffImage_HR_i);
		writer->Update();

		std::cout << "Diff " << i << std::endl;

		DiffImageList.push_back(diffImage_HR_i);
	
		}

	return DiffImageList;

}


JointTensorEstimation::ImageListType JointTensorEstimation::ComputeDifferenceImages(TensorImageType::Pointer logTensorImage)
{

	int numOfImages = m_DWIListLR.size();


	TensorUtilities utilsTensor;
	CopyImage cpImage;
	TensorImageType::Pointer tensorImage = utilsTensor.ExpTensorImageFilter (logTensorImage,m_HRmask);

	//Compute LR PredictedImage

	ImageListType DiffImageList;

	for (int i=0; i < numOfImages ; i++)
	{
		ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
		ScalarImageIterator itB0(m_B0Image_HR, m_B0Image_HR->GetLargestPossibleRegion());
		TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());

		ScalarImageType::Pointer predImage_HR_i = ScalarImageType::New();
		cpImage.CopyScalarImage(m_B0Image_HR, predImage_HR_i);
		ScalarImageIterator itPred(predImage_HR_i, predImage_HR_i->GetLargestPossibleRegion());

		ScalarImageType::Pointer attenImage_i = ScalarImageType::New();
		cpImage.CopyScalarImage(m_B0Image_HR, attenImage_i);

		for (itPred.GoToBegin(), itHRMask.GoToBegin(), itB0.GoToBegin(), itTens.GoToBegin();
				!itPred.IsAtEnd(), !itHRMask.IsAtEnd(), !itB0.IsAtEnd(), !itTens.IsAtEnd();
				++itPred, ++itHRMask, ++itB0, ++itTens)
		{
		   if (itHRMask.Get() != 0)
		   {

		  RealType atten_i;
		  vnl_vector<double> g_temp_i= m_GradList[i]->GetPixel(itHRMask.GetIndex()).GetVnlVector();
		  vnl_vector<RealType> g_i; g_i.set_size(3);
		  vnl_copy(g_temp_i, g_i);
		
	//	  std::cout << g_temp_i << std::endl;
		
		vnl_matrix<RealType> g_mat_i;
		g_mat_i.set_size(3,1);
		g_mat_i.set_column(0,g_i);
		DiffusionTensorType D = itTens.Get();
		MatrixType D_mat;
		D_mat.set_size(3,3);
		D_mat = utilsTensor.ConvertDT2Mat(D);
		 MatrixType temp; temp.set_size(1,1);
		temp = g_mat_i.transpose()*D_mat*g_mat_i;

		atten_i = exp(temp(0,0)*(-1)*m_bval)*itB0.Get();

		attenImage_i->SetPixel(itHRMask.GetIndex(), atten_i);
		itPred.Set(atten_i);

		   }
		}

	//	std::cout << "Atten " << i << std::endl;
		//AttenuationList.push_back(attenImage_i);

		ComposeImageFilter composeFilter;
		composeFilter.GetHRImage(predImage_HR_i);
		composeFilter.GetLRImage(m_LRmask);
		composeFilter.ReadMatrix(m_MapHR2LR);

		ScalarImageType::Pointer predImage_LR_i = composeFilter.ComposeIt();


		SubtractImageFilterType::Pointer subtractImageFilter = SubtractImageFilterType::New();
		subtractImageFilter->SetInput1(predImage_LR_i);
		subtractImageFilter->SetInput2(m_DWIListLR[i]);
		subtractImageFilter->Update();

		ScalarImageType::Pointer diffImage_LR_i = subtractImageFilter->GetOutput();
		diffImage_LR_i->DisconnectPipeline();

		ComposeImageFilter composeFilter2;
		composeFilter2.GetHRImage(diffImage_LR_i);
		composeFilter2.GetLRImage(m_HRmask);
		composeFilter2.ReadMatrix(m_MapLR2HR);

		ScalarImageType::Pointer diffImage_HR_i = composeFilter2.ComposeIt();
		DiffImageList.push_back(diffImage_HR_i);

		 int num =i;
                 std::ostringstream num_con;
                 num_con << num;
                 std::string result  = num_con.str();
		
		std::string Pred_name = "Pred_" + result + ".nii.gz";
		std::string Obs_name = "Obs_" + result + ".nii.gz";
		std::string Diff_name = "Diff_" + result + ".nii.gz";		
	
		ScalarWriterType::Pointer scalarWriter_pred = ScalarWriterType::New();
		ScalarWriterType::Pointer scalarWriter_Obs = ScalarWriterType::New();
		ScalarWriterType::Pointer scalarWriter_Diff = ScalarWriterType::New();		

		scalarWriter_pred->SetFileName(Pred_name.c_str());
		scalarWriter_pred->SetInput(predImage_LR_i);
		scalarWriter_pred->Update();

		scalarWriter_Obs->SetFileName(Obs_name.c_str());
		scalarWriter_Obs->SetInput(m_DWIListLR[i]);
		scalarWriter_Obs->Update();

		scalarWriter_Diff->SetFileName(Diff_name.c_str());
		scalarWriter_Diff->SetInput(diffImage_LR_i);
		scalarWriter_Diff->Update();
	
	
}


	return DiffImageList;
	//Do the check
}



JointTensorEstimation::ImageListType JointTensorEstimation::ComputeAttenuation_Frac(TensorImageType::Pointer logTensorImage)
{
	int numOfImages= m_DWIListHR.size();
	TensorUtilities utilsTensor;
	CopyImage cpImage;
	TensorImageType::Pointer tensorImage = utilsTensor.ExpTensorImageFilter(logTensorImage, m_HRmask);

	ImageListType AttenuationImageList;

	std::cout << numOfImages << std::endl;

	for (int i=0; i < numOfImages; i++)
	{
		ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
		ScalarImageIterator itB0(m_B0Image_HR, m_B0Image_HR->GetLargestPossibleRegion());
		TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());

		ScalarImageType::Pointer attenImage_i = ScalarImageType::New();
		cpImage.CopyScalarImage(m_B0Image_HR, attenImage_i);

		for ( itHRMask.GoToBegin(), itB0.GoToBegin(), itTens.GoToBegin(); !itHRMask.IsAtEnd(), !itB0.IsAtEnd(), !itTens.IsAtEnd();  ++itHRMask, ++itB0, ++itTens)
		{
		if (itHRMask.Get() != 0)
		{
		RealType atten_i;
		vnl_vector<double> g_temp_i= m_GradList[i]->GetPixel(itHRMask.GetIndex()).GetVnlVector();
		vnl_vector<RealType> g_i; g_i.set_size(3);
		vnl_copy(g_temp_i, g_i);


		 vnl_matrix<RealType> g_mat_i;
                 g_mat_i.set_size(3,1);
                 g_mat_i.set_column(0,g_i);
                 DiffusionTensorType D = itTens.Get();
                 MatrixType D_mat;
                 D_mat.set_size(3,3);
                 D_mat = utilsTensor.ConvertDT2Mat(D);
                 MatrixType temp; temp.set_size(1,1);
                 temp = g_mat_i.transpose()*D_mat*g_mat_i;

                 atten_i = exp(temp(0,0)*(-1)*m_bval);

		attenImage_i->SetPixel(itHRMask.GetIndex(), atten_i);

		}


	}

		typedef itk::ImageFileWriter<ScalarImageType> ScalarImageWriterType;
		ScalarImageWriterType::Pointer scalarImageWriter = ScalarImageWriterType::New();

	         int num =i;
                 std::ostringstream num_con;
	         num_con << num; 
                 std::string result  = num_con.str();
		
		 std::string Pred_name = "Atten_" + result + ".nii.gz";
		
		scalarImageWriter->SetFileName(Pred_name);
		scalarImageWriter->SetInput(attenImage_i);
		scalarImageWriter->Update();


		std::cout << "Atten " << i << std::endl;	


		AttenuationImageList.push_back(attenImage_i);

	}




	return AttenuationImageList;
}


JointTensorEstimation::ImageListType JointTensorEstimation::ComputeAttenuation(TensorImageType::Pointer logTensorImage)
{
	int numOfImages= m_DWIListLR.size();
	TensorUtilities utilsTensor;
	CopyImage cpImage;
	TensorImageType::Pointer tensorImage = utilsTensor.ExpTensorImageFilter(logTensorImage, m_HRmask);

	ImageListType AttenuationImageList;
        
	for (int i=0; i < numOfImages; i++)
	{
		ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
		ScalarImageIterator itB0(m_B0Image_HR, m_B0Image_HR->GetLargestPossibleRegion());
		TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());
	
		ScalarImageType::Pointer attenImage_i = ScalarImageType::New();
		cpImage.CopyScalarImage(m_B0Image_HR, attenImage_i);
		
		for ( itHRMask.GoToBegin(), itB0.GoToBegin(), itTens.GoToBegin(); !itHRMask.IsAtEnd(), !itB0.IsAtEnd(), !itTens.IsAtEnd();  ++itHRMask, ++itB0, ++itTens)
		{
		if (itHRMask.Get() != 0)
		{	
		RealType atten_i;
		vnl_vector<double> g_temp_i= m_GradList[i]->GetPixel(itHRMask.GetIndex()).GetVnlVector();
		vnl_vector<RealType> g_i; g_i.set_size(3);
		vnl_copy(g_temp_i, g_i);


		 vnl_matrix<RealType> g_mat_i;
                 g_mat_i.set_size(3,1);
                 g_mat_i.set_column(0,g_i);
                 DiffusionTensorType D = itTens.Get();
                 MatrixType D_mat;
                 D_mat.set_size(3,3);
                 D_mat = utilsTensor.ConvertDT2Mat(D);
                 MatrixType temp; temp.set_size(1,1);
                 temp = g_mat_i.transpose()*D_mat*g_mat_i;

                 atten_i = exp(temp(0,0)*(-1)*m_bval)*itB0.Get();

		attenImage_i->SetPixel(itHRMask.GetIndex(), atten_i);

		}
	}

		AttenuationImageList.push_back(attenImage_i);

	}

	return AttenuationImageList;
}



JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::ComputeDelSim_Frac_DispField(TensorImageType::Pointer logTensorImage)
{
	TensorImageType::Pointer delTotalSim = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(m_dt_init, delTotalSim);

	int numOfImages = m_DWIListHR.size();
	TensorUtilities utilsTensor;
	TensorImageType::Pointer tensorImage = utilsTensor.ExpTensorImageFilter(logTensorImage, m_HRmask);

	ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	TensorImageIterator itDelSimTotal(delTotalSim, delTotalSim->GetLargestPossibleRegion());
	TensorImageIterator itLogTens(logTensorImage, logTensorImage->GetLargestPossibleRegion());

	TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());
	ScalarImageIterator itB0(m_B0Image_HR, m_B0Image_HR->GetLargestPossibleRegion());

	ImageListType DiffImageList, AttenuationList;
	DiffImageList = ComputeDifferenceImages_Frac(logTensorImage);
	AttenuationList = ComputeAttenuation_Frac(logTensorImage);

	ScalarImageType::IndexType IndexG, IndexB;
	IndexB[0]=123; IndexB[1]=159; IndexB[2]=201;
	IndexG[0]=123; IndexG[1]=138; IndexG[2]=171;


	
	for (itB0.GoToBegin(), itTens.GoToBegin(), itDelSimTotal.GoToBegin(), itHRMask.GoToBegin(); !itB0.IsAtEnd(), !itTens.IsAtEnd(), !itDelSimTotal.IsAtEnd(), !itHRMask.IsAtEnd(); ++itB0, ++itTens, ++itDelSimTotal, ++itHRMask)
	{
	 if (itHRMask.Get() != 0)
	{
		DiffusionTensorType totalDelSim;
		for (int i =0; i < numOfImages; i++)
		{

//			std::cout << "Image " << i << std::endl;
				DiffusionTensorType delG_ExpL;

				delG_ExpL = utilsTensor.MatrixExpDirDerivative(itLogTens.Get(), m_GradList[i]->GetPixel(itHRMask.GetIndex()), itHRMask.GetIndex());


				DiffusionTensorType delSim_DelL_i;
				delSim_DelL_i = delG_ExpL*(DiffImageList[i]->GetPixel(itHRMask.GetIndex())*(-m_bval)*AttenuationList[i]->GetPixel(itHRMask.GetIndex()));

				totalDelSim = totalDelSim + delSim_DelL_i;

		}

		itDelSimTotal.Set(totalDelSim);
	}

	}

	std::cout << "DelTotalSim " << std::endl;

	return delTotalSim;
}




JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::ComputeDelSim_DispField(TensorImageType::Pointer logTensorImage)
{
	TensorImageType::Pointer delTotalSim = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(m_dt_init, delTotalSim);

       TensorImageType::IndexType IndexG, IndexB;
       IndexG[0]=123; IndexG[1]=138; IndexG[2]=171;
       IndexB[0]=123; IndexB[1]=159; IndexB[2]=201;



	int numOfImages = m_DWIListLR.size();
	TensorUtilities utilsTensor;
	TensorImageType::Pointer tensorImage = utilsTensor.ExpTensorImageFilter(logTensorImage, m_HRmask);
	
	ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	TensorImageIterator itDelSimTotal(delTotalSim, delTotalSim->GetLargestPossibleRegion());
	TensorImageIterator itLogTens(logTensorImage, logTensorImage->GetLargestPossibleRegion());

	TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());
	ScalarImageIterator itB0(m_B0Image_HR, m_B0Image_HR->GetLargestPossibleRegion());

	ImageListType DiffImageList, AttenuationList;
	DiffImageList = ComputeDifferenceImages(logTensorImage);
	AttenuationList = ComputeAttenuation(logTensorImage);

	for (itB0.GoToBegin(), itTens.GoToBegin(), itDelSimTotal.GoToBegin(), itHRMask.GoToBegin();
			!itB0.IsAtEnd(), !itTens.IsAtEnd(), !itDelSimTotal.IsAtEnd(), !itHRMask.IsAtEnd();
			++itB0, ++itTens, ++itDelSimTotal, ++itHRMask)
	{
	 if (itHRMask.Get() != 0)
	{
		DiffusionTensorType totalDelSim;
		for (int i =0; i < numOfImages; i++)
		{

        		DiffusionTensorType delG_ExpL;
				delG_ExpL = utilsTensor.MatrixExpDirDerivative(itLogTens.Get(), m_GradList[i]->GetPixel(itHRMask.GetIndex()), itHRMask.GetIndex());
				DiffusionTensorType delSim_DelL_i;
				delSim_DelL_i = delG_ExpL*(DiffImageList[i]->GetPixel(itHRMask.GetIndex())*(-m_bval)*AttenuationList[i]->GetPixel(itHRMask.GetIndex()));
				totalDelSim = totalDelSim + delSim_DelL_i;
		
/*		   if (itHRMask.GetIndex() == IndexG)
		   {	std::cout << "IndexG" << std::endl;
			std::cout << "delG_ExpL " << delG_ExpL << std::endl; 
			}
		   if (itHRMask.GetIndex() == IndexB)
		   {	std::cout << "IndexB" << std::endl;
			std::cout << "delG_ExpL " << delG_ExpL << std::endl;
		 }
*/ 

		}
		
		itDelSimTotal.Set(totalDelSim);	
	}
	
	}



/*		for (int i=0; i < numOfImages; i++)
		{
			std::cout << "Attenuation " << i << " Good Tensor " << AttenuationList[i]->GetPixel(IndexG) << std::endl;
			std::cout << "Attenuation " << i << "  Bad Tensor " << AttenuationList[i]->GetPixel(IndexB) << std::endl;
			
			std::cout << "Diff Image List " << i << "Good Tensor" << DiffImageList[i]->GetPixel(IndexG) << std::endl;
			std::cout << "Diff Image List " << i << " Badd Tensor" << DiffImageList[i]->GetPixel(IndexB) << std::endl;
		}
*/

	return delTotalSim;	
}



JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::ComputeDelReg(TensorImageType::Pointer FirstTerm, TensorImageType::Pointer SecTerm)
{


	typedef itk::AddImageFilter<TensorImageType, TensorImageType, TensorImageType> AddTensorImageFilterType;
	AddTensorImageFilterType::Pointer filter = AddTensorImageFilterType::New();

	filter->SetInput1(FirstTerm);
	filter->SetInput2(SecTerm);
	filter->Update();
	TensorImageType::Pointer delRegTerm = filter->GetOutput();
	return delRegTerm;

}

JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::ComputeSecondTermDelReg(ScalarImageType::Pointer PsiImage, TensorImageType::Pointer LogTensorImage)
{
	ScalarImageType::SizeType radiusS; radiusS.Fill(1);
	TensorImageType::SizeType radiusT; radiusT.Fill(1);
	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	ScalarNeighIterator itPsi(radiusS, PsiImage, PsiImage->GetLargestPossibleRegion());
	TensorNeighIterator itLogTens(radiusT, LogTensorImage, LogTensorImage->GetLargestPossibleRegion());


	TensorImageType::Pointer secondTerm = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(LogTensorImage, secondTerm);

	ScalarImageType::SpacingType spacingS = PsiImage->GetSpacing();
	TensorImageType::SpacingType spacingT = LogTensorImage->GetSpacing();

	   TensorImageType::IndexType testIdx, testIdx1;
	   testIdx[0]=11; testIdx[1]=21; testIdx[2] = 0;


	TensorImageIterator itTens(secondTerm, secondTerm->GetLargestPossibleRegion());

	for (itTens.GoToBegin(), itMask.GoToBegin(), itPsi.GoToBegin(), itLogTens.GoToBegin();
			!itTens.IsAtEnd(), !itMask.IsAtEnd(), !itPsi.IsAtEnd(), !itLogTens.IsAtEnd();
			++itTens, ++itMask, ++itPsi, ++itLogTens)
	{
		if (itMask.Get() != 0)
		{


			DiffusionTensorType TotalSecTerm;
			for (int i=0; i < 3 ; i++)
			{
				//Partial Psi
				RealType partialPsi = (itPsi.GetNext(i) - itPsi.GetPrevious(i))/(2*spacingS[i]);

				//Partial L
				DiffusionTensorType partialL = (itLogTens.GetNext(i) - itLogTens.GetPrevious(i))/(2*spacingT[i]);

				DiffusionTensorType tempD;
				tempD = partialL*partialPsi;

				TotalSecTerm =  TotalSecTerm + tempD;
			}

			itTens.Set(TotalSecTerm*(-2));
		}
	}
	std::cout << "DelReg Second Term" << std::endl;
	return secondTerm;
}

JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::ComputeFirstTermDelReg(ScalarImageType::Pointer PsiImage,
							TensorImageType::Pointer LaplaceTensorImage)
{
	TensorImageType::Pointer FirstTerm = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(LaplaceTensorImage, FirstTerm);

	ScalarImageIterator itPsi(PsiImage, PsiImage->GetLargestPossibleRegion());
	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	TensorImageIterator itLap(LaplaceTensorImage, LaplaceTensorImage->GetLargestPossibleRegion());
	TensorImageIterator itFirst(FirstTerm, FirstTerm->GetLargestPossibleRegion());

	for (itMask.GoToBegin(), itLap.GoToBegin(), itFirst.GoToBegin(), itPsi.GoToBegin();
			!itMask.IsAtEnd(), !itLap.IsAtEnd(), !itFirst.IsAtEnd(), !itPsi.IsAtEnd();
						++itMask, ++itLap, ++itFirst, ++itPsi)
	{
		if (itMask.Get() != 0)
		{
			DiffusionTensorType temp;
			temp = itLap.Get()*(-2)*itPsi.Get();
			itFirst.Set(temp);
		}
	}

	return FirstTerm;

}

JointTensorEstimation::ScalarImageType::Pointer JointTensorEstimation::GradientLogMagTensorImage(TensorImageType::Pointer logTensorImage)
{
	ScalarImageType::Pointer gradMagTensorImage = ScalarImageType::New();

	CopyImage cpImage;
	cpImage.CopyScalarImage(m_HRmask, gradMagTensorImage);

	TensorImageType::SizeType radius; radius.Fill(1);

	TensorUtilities utilsTens;
//	TensorImageType::Pointer logTensorImage = utilsTens.LogTensorImageFilter(m_dt_init, m_maskImage);

	TensorImageType::IndexType testIdx;
	testIdx[0]=11; testIdx[1]=22; testIdx[2]=0;



	ScalarImageIterator itGradMag(gradMagTensorImage, gradMagTensorImage->GetLargestPossibleRegion());
	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	TensorImageIterator itTens(logTensorImage, logTensorImage->GetLargestPossibleRegion());
	TensorNeighIterator itNeighTens(radius, logTensorImage, logTensorImage->GetLargestPossibleRegion());

	TensorImageType::SpacingType space;
	space=logTensorImage->GetSpacing();

	for (itTens.GoToBegin(), itMask.GoToBegin(), itGradMag.GoToBegin(), itNeighTens.GoToBegin();
			!itTens.IsAtEnd(), !itMask.IsAtEnd(), !itGradMag.IsAtEnd(), !itNeighTens.IsAtEnd();
			++itTens, ++itMask, ++itGradMag, ++itNeighTens)
	{

		if (itMask.Get() !=0)
		{
			RealType total=0;
			for (int i=0; i< 3; i++)
			{
				DiffusionTensorType temp = (itNeighTens.GetNext(i) - itNeighTens.GetPrevious(i))/(2*space[i]);
				RealType LENorm = utilsTens.ComputeTensorNorm(temp);
				RealType LENormSquared = LENorm*LENorm;
				total = total + LENormSquared;

			}
                        itGradMag.Set(sqrt(total)); // ||delta L ||

                        if (isnan(total) == 1)
                        	std::cout << itMask.GetIndex() << std::endl;
		}
	}

	return gradMagTensorImage;

}

JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::ComputeLaplaceTensor(TensorImageType::Pointer LogTensorImage)
{
	CopyImage cpImage;
	TensorImageType::Pointer LaplaceTensorImage = TensorImageType::New();

	TensorUtilities utilsTensor;
//	TensorImageType::Pointer LogTensorImage = utilsTensor.LogTensorImageFilter(m_dt_init, m_maskImage);

	cpImage.CopyTensorImage(LogTensorImage, LaplaceTensorImage);

	TensorImageType::SpacingType spacing = LogTensorImage->GetSpacing();
	TensorImageType::SizeType radius; radius.Fill(1);

	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	TensorNeighIterator itLogTens(radius, LogTensorImage,LogTensorImage->GetLargestPossibleRegion());
	TensorImageIterator itLapTens(LaplaceTensorImage, LaplaceTensorImage->GetLargestPossibleRegion());

	for(itMask.GoToBegin(), itLogTens.GoToBegin(), itLapTens.GoToBegin();
			!itMask.IsAtEnd(), !itLogTens.IsAtEnd(), !itLapTens.IsAtEnd();
			++itMask, ++itLogTens, ++itLapTens)
	{
		if (itMask.Get() != 0)
		{
			DiffusionTensorType LogTotal;
			for (int i=0; i < 3; i++)
			{
				DiffusionTensorType LogDir;
				LogDir = (itLogTens.GetNext(i) -itLogTens.GetCenterPixel()*2 + itLogTens.GetPrevious(i))/(spacing[i]*spacing[i]);
				LogTotal = LogTotal + LogDir;
			}
			itLapTens.Set(LogTotal);
		}
	}

	return LaplaceTensorImage;


}

JointTensorEstimation::ScalarImageType::Pointer JointTensorEstimation::ComputePsiImage(ScalarImageType::Pointer gradMagImage)
{
	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	ScalarImageIterator itGradMag(gradMagImage, gradMagImage->GetLargestPossibleRegion());

	ScalarImageType::Pointer psiImage = ScalarImageType::New();
	CopyImage cpImage;
	cpImage.CopyScalarImage(gradMagImage, psiImage);

	ScalarWriterType::Pointer writer = ScalarWriterType::New();
	writer->SetFileName("GradMagTensorImage1.nii.gz");
	writer->SetInput(gradMagImage);
	writer->Update();

	ScalarImageIterator itPsi(psiImage, psiImage->GetLargestPossibleRegion());

	for (itPsi.GoToBegin(), itMask.GoToBegin(), itGradMag.GoToBegin();
			!itPsi.IsAtEnd(), !itMask.IsAtEnd(), !itGradMag.IsAtEnd();
			++itPsi, ++itMask, ++itGradMag)
	{
		if (itMask.Get() != 0)
		{

			RealType temp = itGradMag.Get()/m_kappa;
			RealType psiTot = 1/sqrt(1+temp*temp);
			itPsi.Set(psiTot);
		}
	}

	return psiImage;

}

JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::UpdateTerms1()
{
	std::cout << "Update " << std::endl;
	
	TensorUtilities utils;
        std::vector<RealType> Energy_vec;
 
	 CopyImage cpImage;

	TensorImageType::Pointer log_tensorImage_init = utils.LogTensorImageFilter(m_dt_init, m_HRmask);

	ImageListType PredImageList = ComputeAttenuation_Frac(log_tensorImage_init);
	
	std::cout << "COmputed predicted ImageList " << std::endl;
	TensorImageType::Pointer logTensorImage_n_1 = TensorImageType::New();
	cpImage.CopyTensorImage(m_dt_init, logTensorImage_n_1);


	//Compute TotalEnergy
         TotalEnergy totEnergyComputation;
	totEnergyComputation.ReadObsImageList(m_DWIListHR);
        totEnergyComputation.ReadPredImageList(PredImageList);
        totEnergyComputation.ReadSigma(m_Sigma);
        totEnergyComputation.ReadKappa(m_kappa);
        totEnergyComputation.ReadHRMaskImage(m_HRmask);
        totEnergyComputation.ReadB0Image(m_B0Image_HR);

     //Compute Regularization Energy
          RealType regEnergy_0 =0;
 //      regEnergy_0 = totEnergyComputation.RegularizationEnergy(log_tensorImage_init);
 //      std::cout << "RegEnergy Computed " << std::endl;
 
         RealType GaussEnergy_0 =0;
         GaussEnergy_0 = totEnergyComputation.GaussianNoise_woSR(log_tensorImage_init);
	
	 std::cout << "GaussEnergy Computed " << std::endl;
	
	RealType totalEnergy_0 =0;
 
        totalEnergy_0 = GaussEnergy_0 + m_Lambda*regEnergy_0;
 
          std::cout << fixed;
          std::cout << "Gauss Energy 0 " << GaussEnergy_0 << std::endl;
          std::cout << "Reg Energy 0 " << regEnergy_0 << std::endl;
 
	Energy_vec.push_back(totalEnergy_0);
	ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	TensorImageType::Pointer logTensorImage_n = log_tensorImage_init;        
	
	 for (int i =0; i < m_Iterations; i++)
	{
	    TensorImageType::Pointer delSim = TensorImageType::New();
           cpImage.CopyTensorImage(logTensorImage_n, delSim);
           delSim = ComputeDelSim_Frac_DispField(logTensorImage_n);
	
	   std::cout << "Computed Del Sim Frac " << std::endl;

   	  TensorImageType::Pointer delRegTerm = TensorImageType::New();
 	  cpImage.CopyTensorImage(delSim, delRegTerm);
 

/*	   ScalarImageType::Pointer gradMagTensorImage = GradientLogMagTensorImage(logTensorImage_n);
 
        ScalarImageType::Pointer psiImage = ComputePsiImage(gradMagTensorImage);
        TensorImageType::Pointer LaplaceImage = ComputeLaplaceTensor(logTensorImage_n);
          TensorImageType::Pointer FirstTerm = ComputeFirstTermDelReg(psiImage, logTensorImage_n);
         TensorImageType::Pointer SecTerm = ComputeSecondTermDelReg(psiImage, logTensorImage_n);
           delRegTerm = ComputeDelReg(FirstTerm, SecTerm);

*/

	TensorImageIterator itDelSim(delSim, delSim->GetLargestPossibleRegion());
        TensorImageIterator itDelReg(delRegTerm, delRegTerm->GetLargestPossibleRegion());
        TensorImageIterator itL_n(logTensorImage_n, logTensorImage_n->GetLargestPossibleRegion());

        TensorImageIterator itL_n_1(logTensorImage_n_1, logTensorImage_n_1->GetLargestPossibleRegion());
	

	
         for (itHRMask.GoToBegin(), itDelSim.GoToBegin(), itDelReg.GoToBegin(), itL_n.GoToBegin(), itL_n_1.GoToBegin();
         !itHRMask.IsAtEnd(), !itDelSim.IsAtEnd(), !itDelReg.IsAtEnd(), !itL_n.IsAtEnd(), !itL_n_1.IsAtEnd();
         ++itHRMask, ++itDelSim, ++itDelReg, ++itL_n, ++itL_n_1)
         {
                 if (itHRMask.Get() != 0)
                 {
                 DiffusionTensorType L_temp;
                 L_temp = itL_n.Get() + (itDelSim.Get() + itDelReg.Get()*m_Lambda)*m_Step_size;

                 RealType Log_Trace = L_temp.GetTrace();

                 itL_n_1.Set(L_temp);

                 }

        }

         TensorUtilities utilsTensor1;
         std::cout << "Taking Exponents " << std::endl;
         TensorImageType::Pointer tensorImage_n_1 = utilsTensor1.ExpTensorImageFilter(logTensorImage_n_1, m_HRmask);

         std::cout << "Took Exponent " << std::endl;

         typedef itk::ImageFileWriter<TensorImageType> TensorImageWriterType;
         TensorImageWriterType::Pointer tensorImageWriter = TensorImageWriterType::New();
         tensorImageWriter->SetFileName("AfterTakingExp.nii.gz");
         tensorImageWriter->SetInput(tensorImage_n_1);
         tensorImageWriter->Update();

         TensorImageType::Pointer removedNansInfs_TensorImage = utilsTensor1.ReplaceNaNsInfsExpTensor(tensorImage_n_1, m_HRmask);
         TensorImageWriterType::Pointer tensorImageWriter1 = TensorImageWriterType::New();
         tensorImageWriter1->SetFileName("AfterTakingExp_removed.nii.gz");
         tensorImageWriter1->SetInput(removedNansInfs_TensorImage);
         tensorImageWriter1->Update();

         TensorImageType::Pointer recomputeLog_n_1 = utilsTensor1.LogTensorImageFilter(removedNansInfs_TensorImage, m_HRmask);

         std::cout << "Recomputed Log " << std::endl;
                 TensorImageType::Pointer removed_nans_logTensorImage = utilsTensor1.ReplaceNaNsInfs(recomputeLog_n_1, m_HRmask);
         std::cout << "Removed Nans " << std::endl;

//   TotalEnergy
	PredImageList = ComputeAttenuation_Frac(removed_nans_logTensorImage);
        totEnergyComputation.ReadPredImageList(PredImageList);	

         RealType gaussEnergy_n_1, regEnergy_n_1;
         gaussEnergy_n_1= 0; regEnergy_n_1 =0;
         gaussEnergy_n_1 = totEnergyComputation.GaussianNoise_woSR(removed_nans_logTensorImage);
         //regEnergy_n_1 = totEnergyComputation.RegularizationEnergy(removed_nans_logTensorImage);

         RealType energy_n_1 = gaussEnergy_n_1 + m_Lambda*regEnergy_n_1 ;

         std::cout << fixed;
         std::cout << "Energy " << Energy_vec.back() << std::endl;
         std::cout << "New Energy " << energy_n_1 << std::endl;


         if (energy_n_1 < Energy_vec.back())
        {
                 logTensorImage_n = removed_nans_logTensorImage;
                 Energy_vec.push_back(energy_n_1);

         }
        else
         { break;  }


	}

	TensorImageType::Pointer tensorImage = utils.ExpTensorImageFilter(logTensorImage_n, m_HRmask);

	return tensorImage;
	
}

