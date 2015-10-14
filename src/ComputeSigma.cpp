/*
 * ComputeSigma.cxx
 *
 *  Created on: Aug 2, 2015
 *      Author: vgupta
 */

#include "ComputeSigma.h"

void ComputeSigma::ReadDWIList(ImageListType listImage)
{
	m_DWIList = listImage;
}

void ComputeSigma::ReadMaskImage(ScalarImageType::Pointer maskImage)
{
	m_maskImage = maskImage;
}

void ComputeSigma::ReadGradientList(VectorImageListType gradList)
{
	m_GradList = gradList;
}

void ComputeSigma::ReadTensorImage(TensorImageType::Pointer tensorImage)
{
	m_dt = tensorImage;
}

void ComputeSigma::ReadBVal(RealType bval)
{
	m_BVal = bval;
}

void ComputeSigma::ReadB0Image(ScalarImageType::Pointer image)
{
	m_B0Image = image;
}

vnl_vector<ComputeSigma::RealType> ComputeSigma::ComputeAttenuation_woSR()
{
	int numOfImages= m_DWIList.size();
	int numOfGrads = m_GradList.size();

	ScalarImageType::Pointer m_B0 = m_DWIList[0];

	TensorUtilities tensUtilties;
	CopyImage cpImage;

	vnl_vector<RealType> Sigma;
	Sigma.set_size(numOfGrads);

	for (int i=0; i < numOfGrads ; i++)
	{

		ScalarImageType::Pointer Atten_im = ScalarImageType::New();
		cpImage.CopyScalarImage(m_maskImage, Atten_im);


		ScalarImageIterator itAtten(Atten_im, Atten_im->GetLargestPossibleRegion());
		ScalarImageIterator itDWI(m_DWIList[i], m_DWIList[i]->GetLargestPossibleRegion());
		TensorImageIterator itTens(m_dt, m_dt->GetLargestPossibleRegion());
		VectorImageIterator itGrad(m_GradList[i], m_GradList[i]->GetLargestPossibleRegion());
		ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
		ScalarImageIterator itB0(m_B0Image, m_B0Image->GetLargestPossibleRegion());

		SubtractImageFilterType::Pointer subImageFilter = SubtractImageFilterType::New();
		StatisticsImageFilterType::Pointer statisticImageFilter = StatisticsImageFilterType::New();

		ScalarImageType::Pointer diffImage_i = ScalarImageType::New();
		cpImage.CopyScalarImage(m_B0Image, diffImage_i);


		for (itDWI.GoToBegin(), itTens.GoToBegin(), itGrad.GoToBegin(), itMask.GoToBegin(), itAtten.GoToBegin(), itB0.GoToBegin();
				!itDWI.IsAtEnd(), !itTens.IsAtEnd(), !itGrad.IsAtEnd(), !itMask.IsAtEnd(), !itAtten.IsAtEnd(), !itB0.IsAtEnd();
				++itDWI, ++itTens, ++itGrad, ++itMask, ++itAtten, ++itB0)
		{
			if (itMask.Get() != 0)
			{

				RealType Atten_i_val;
				vnl_vector<RealType> g_i = itGrad.Get().GetVnlVector();
				vnl_matrix<RealType> g_mat_i;
				g_mat_i.set_size(3,1);
				g_mat_i.set_column(0,g_i);

				DiffusionTensorType D = itTens.Get();
				MatrixType D_mat;
				D_mat.set_size(3,3);
				D_mat = tensUtilties.ConvertDT2Mat(D);

				MatrixType temp; temp.set_size(1,1);
				temp = g_mat_i.transpose()*D_mat*g_mat_i;

				Atten_i_val = exp(-m_BVal*temp(0,0));
				itAtten.Set(Atten_i_val);

				RealType diff = itDWI.Get()/itB0.Get() - Atten_i_val;
				
		
				diffImage_i->SetPixel(itMask.GetIndex(), diff);		

		}
		}


		statisticImageFilter->SetInput(diffImage_i);
		statisticImageFilter->Update();


		RealType sigma = statisticImageFilter->GetSigma();
		Sigma.put(i, sigma);

	}

	return Sigma;

}

vnl_vector<ComputeSigma::RealType> ComputeSigma::ComputeAttenuation()
{
	int numOfImages= m_DWIList.size();
	int numOfGrads = m_GradList.size();

	ScalarImageType::Pointer m_B0 = m_DWIList[0];

	TensorUtilities tensUtilties;
	CopyImage cpImage;

	vnl_vector<RealType> Sigma;
	Sigma.set_size(numOfGrads);

	for (int i=0; i < numOfGrads ; i++)
	{

		ScalarImageType::Pointer Atten_im = ScalarImageType::New();
		cpImage.CopyScalarImage(m_maskImage, Atten_im);


		ScalarImageIterator itAtten(Atten_im, Atten_im->GetLargestPossibleRegion());
		ScalarImageIterator itDWI(m_DWIList[i], m_DWIList[i]->GetLargestPossibleRegion());
		TensorImageIterator itTens(m_dt, m_dt->GetLargestPossibleRegion());
		VectorImageIterator itGrad(m_GradList[i], m_GradList[i]->GetLargestPossibleRegion());
		ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
		ScalarImageIterator itB0(m_B0Image, m_B0Image->GetLargestPossibleRegion());

		SubtractImageFilterType::Pointer subImageFilter = SubtractImageFilterType::New();
		StatisticsImageFilterType::Pointer statisticImageFilter = StatisticsImageFilterType::New();


		for (itDWI.GoToBegin(), itTens.GoToBegin(), itGrad.GoToBegin(), itMask.GoToBegin(), itAtten.GoToBegin(), itB0.GoToBegin();
				!itDWI.IsAtEnd(), !itTens.IsAtEnd(), !itGrad.IsAtEnd(), !itMask.IsAtEnd(), !itAtten.IsAtEnd(), !itB0.IsAtEnd();
				++itDWI, ++itTens, ++itGrad, ++itMask, ++itAtten, ++itB0)
		{
			if (itMask.Get() != 0)
			{

				RealType Atten_i_val;
				vnl_vector<RealType> g_i = itGrad.Get().GetVnlVector();
				vnl_matrix<RealType> g_mat_i;
				g_mat_i.set_size(3,1);
				g_mat_i.set_column(0,g_i);

				DiffusionTensorType D = itTens.Get();
				MatrixType D_mat;
				D_mat.set_size(3,3);
				D_mat = tensUtilties.ConvertDT2Mat(D);

				MatrixType temp; temp.set_size(1,1);
				temp = g_mat_i.transpose()*D_mat*g_mat_i;

				Atten_i_val = exp(-m_BVal*temp(0,0))*itB0.Get();
				itAtten.Set(Atten_i_val);
		}
		}

		subImageFilter->SetInput1(m_DWIList[i]);
		subImageFilter->SetInput2(Atten_im);
		subImageFilter->Update();

		ScalarImageType::Pointer diffImage = subImageFilter->GetOutput();
		diffImage->DisconnectPipeline();

		statisticImageFilter->SetInput(diffImage);
		statisticImageFilter->Update();


		RealType sigma = statisticImageFilter->GetSigma();
		Sigma.put(i, sigma);

	}

	return Sigma;
}

vnl_vector<ComputeSigma::RealType> ComputeSigma::ComputeAttenuation_Frac()
{
	int numOfImages= m_DWIList.size();
	int numOfGrads = m_GradList.size();

	ScalarImageType::Pointer m_B0 = m_DWIList[0];

	TensorUtilities tensUtilties;
	CopyImage cpImage;

	vnl_vector<RealType> Sigma;
	Sigma.set_size(numOfGrads);

	for (int i=0; i < numOfGrads ; i++)
	{

		ScalarImageType::Pointer Atten_im = ScalarImageType::New();
		cpImage.CopyScalarImage(m_maskImage, Atten_im);


		ScalarImageIterator itAtten(Atten_im, Atten_im->GetLargestPossibleRegion());
		ScalarImageIterator itDWI(m_DWIList[i], m_DWIList[i]->GetLargestPossibleRegion());
		TensorImageIterator itTens(m_dt, m_dt->GetLargestPossibleRegion());
		VectorImageIterator itGrad(m_GradList[i], m_GradList[i]->GetLargestPossibleRegion());
		ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
		ScalarImageIterator itB0(m_B0Image, m_B0Image->GetLargestPossibleRegion());

		SubtractImageFilterType::Pointer subImageFilter = SubtractImageFilterType::New();
		StatisticsImageFilterType::Pointer statisticImageFilter = StatisticsImageFilterType::New();


		for (itDWI.GoToBegin(), itTens.GoToBegin(), itGrad.GoToBegin(), itMask.GoToBegin(), itAtten.GoToBegin(), itB0.GoToBegin();
				!itDWI.IsAtEnd(), !itTens.IsAtEnd(), !itGrad.IsAtEnd(), !itMask.IsAtEnd(), !itAtten.IsAtEnd(), !itB0.IsAtEnd();
				++itDWI, ++itTens, ++itGrad, ++itMask, ++itAtten, ++itB0)
		{
			if (itMask.Get() != 0)
			{

				RealType Atten_i_val;
				vnl_vector<RealType> g_i = itGrad.Get().GetVnlVector();
				vnl_matrix<RealType> g_mat_i;
				g_mat_i.set_size(3,1);
				g_mat_i.set_column(0,g_i);

				DiffusionTensorType D = itTens.Get();
				MatrixType D_mat;
				D_mat.set_size(3,3);
				D_mat = tensUtilties.ConvertDT2Mat(D);

				MatrixType temp; temp.set_size(1,1);
				temp = g_mat_i.transpose()*D_mat*g_mat_i;

				Atten_i_val = exp(-m_BVal*temp(0,0));
				itAtten.Set(Atten_i_val);
		}
		}

//


		DivideByImageFilterType::Pointer divideByImageFilter  = DivideByImageFilterType::New();
		divideByImageFilter->SetInput1(m_DWIList[i]);
		divideByImageFilter->SetInput2(m_B0Image);
		divideByImageFilter->Update();

		ScalarImageType::Pointer Obs_Atten = divideByImageFilter->GetOutput();
		Obs_Atten->DisconnectPipeline();

		subImageFilter->SetInput1(Obs_Atten);
		subImageFilter->SetInput2(Atten_im);
		subImageFilter->Update();

		ScalarImageType::Pointer diffImage = subImageFilter->GetOutput();
		diffImage->DisconnectPipeline();

		statisticImageFilter->SetInput(diffImage);
		statisticImageFilter->Update();


//		int num =i;
//		std::ostringstream num_con;
//		num_con << num;
//		std::string result  = num_con.str();
//
//		std::string tempName = "Diff_" + result + ".nii.gz";
//		std::string tempName1 = "Attn_" + result + ".nii.gz";
//		std::string tempName2 = "ObsAtten_" + result + ".nii.gz";
//
//		WriterType::Pointer writer = WriterType::New();
//		writer->SetFileName(tempName);
//		writer->SetInput(diffImage);
//		writer->Update();
//
//		WriterType::Pointer writer1 = WriterType::New();
//		writer1->SetFileName(tempName1);
//		writer1->SetInput(Atten_im);
//		writer1->Update();
//
//		WriterType::Pointer writer2 = WriterType::New();
//		writer2->SetFileName(tempName2);
//		writer2->SetInput(Obs_Atten);
//		writer2->Update();

		RealType sigma = statisticImageFilter->GetSigma();
		Sigma.put(i, sigma);
//		std::cout << sigma << std::endl;

	}

	return Sigma;
}
