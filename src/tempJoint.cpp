/*
 * JointTensorEstimation.cxx
 *
 *  Created on: Jul 28, 2015
 *      Author: vgupta
 */


#include "math.h"
#include "JointTensorEstimation.h"
#include "itkImageFileWriter.h"
#include "math.h"

using namespace std;
void JointTensorEstimation::ReadDWIList(ImageListType list)
{
	m_DWIList = list;
}

void JointTensorEstimation::ReadGradientList(VectorImageListType vecList)
{
	m_GradList = vecList;
}

void JointTensorEstimation::ReadLamba(RealType lambda)
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

void JointTensorEstimation::ReadMaskImage(ScalarImageType::Pointer maskImage)
{
	m_maskImage = maskImage;
}

void JointTensorEstimation::ReadStepSize(RealType step_size)
{
	m_Step_size =step_size;
}

void JointTensorEstimation::ReadNumOfIterations(int num)
{
	m_Iterations =num;
}

void JointTensorEstimation::ReadB0Image(ScalarImageType::Pointer image)
{
	m_B0Image = image;
}

void JointTensorEstimation::ReadB0Thres(RealType thres)
{
		m_thres = thres;
}

void JointTensorEstimation::ReadMapMatrix(SparseMatrixType map)
{
	m_SpVnl_Row_normalized = map;
}

JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::ComputeDelSim_Frac_Disp_Field(TensorImageType::Pointer logTensorImage)
{
	TensorImageType::Pointer delTotalSim = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(m_dt_init, delTotalSim);

	int numOfImage = m_DWIList.size();
	TensorUtilities utilsTensor;
	TensorImageType::Pointer tensorImage = utilsTensor.ExpTensorImageFilter(logTensorImage, m_maskImage);

	ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
	TensorImageIterator itDelSimTotal(delTotalSim, delTotalSim->GetLargestPossibleRegion());
	TensorImageIterator itLogTens(logTensorImage, logTensorImage->GetLargestPossibleRegion());
	TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());
	ScalarImageIterator itB0(m_B0Image, m_B0Image->GetLargestPossibleRegion());

	for(itMask.GoToBegin(), itDelSimTotal.GoToBegin(), itLogTens.GoToBegin(), itB0.GoToBegin();
			!itMask.IsAtEnd(), !itDelSimTotal.IsAtEnd(), !itLogTens.IsAtEnd(), !itB0.IsAtEnd();
			++itMask, ++itDelSimTotal, ++itLogTens, ++itB0)
	{
		if (itMask.Get() != 0)
		{

		}
	}

}


JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::ComputeDelSim_nonFrac(TensorImageType::Pointer logTensorImage)
{
	TensorImageType::Pointer delTotalSim  = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(m_dt_init, delTotalSim);

	int numOfImages = m_DWIList.size();

	TensorUtilities utilsTensor;
	TensorImageType::Pointer tensorImage = utilsTensor.ExpTensorImageFilter(logTensorImage, m_maskImage);

	ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
	TensorImageIterator itDelSimTotal(delTotalSim, delTotalSim->GetLargestPossibleRegion());
	TensorImageIterator itLogTens(logTensorImage, logTensorImage->GetLargestPossibleRegion());
	TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());
	ScalarImageIterator itB0(m_B0Image, m_B0Image->GetLargestPossibleRegion());

	for (itMask.GoToBegin(), itDelSimTotal.GoToBegin(), itLogTens.GoToBegin(), itB0.GoToBegin();
			!itMask.IsAtEnd(), !itDelSimTotal.IsAtEnd(), !itLogTens.IsAtEnd(), !itB0.IsAtEnd();
			++itMask, ++itDelSimTotal, ++itLogTens, ++itB0)
	{
		if (itMask.Get() != 0)
		{
			DiffusionTensorType totalDelSim;
			for (int i=0; i < numOfImages; i++)
			{
				RealType atten_i;
				vnl_vector<RealType> g_i = m_GradList[i]->GetPixel(itMask.GetIndex()).GetVnlVector();
				vnl_matrix<RealType> g_mat_i;
				g_mat_i.set_size(3,1);
				g_mat_i.set_column(0,g_i);


				DiffusionTensorType D = utilsTensor.ExpM(itLogTens.Get());
				MatrixType D_mat;
				D_mat.set_size(3,3);
				D_mat = utilsTensor.ConvertDT2Mat(D);

				MatrixType temp; temp.set_size(1,1);
				temp = g_mat_i.transpose()*D_mat*g_mat_i;

				atten_i = exp(temp(0,0)*(-1)*m_bval)*itB0.Get();
				RealType obsValue = m_DWIList[i]->GetPixel(itMask.GetIndex());

				RealType diff = (atten_i - obsValue)/(m_Sigma[i]*m_Sigma[i]);

				DiffusionTensorType delG_ExpL;

				delG_ExpL = utilsTensor.MatrixExpDirDerivative(itLogTens.Get(), m_GradList[i]->GetPixel(itMask.GetIndex()), itMask.GetIndex());


				DiffusionTensorType delSim_DelL_i;
				delSim_DelL_i = delG_ExpL*diff*atten_i*(-m_bval);

				totalDelSim = totalDelSim + delSim_DelL_i;

			}

			itDelSimTotal.Set(totalDelSim);
		}
	}


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
	ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
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
	ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
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
	cpImage.CopyScalarImage(m_maskImage, gradMagTensorImage);

	TensorImageType::SizeType radius; radius.Fill(1);

	TensorUtilities utilsTens;
//	TensorImageType::Pointer logTensorImage = utilsTens.LogTensorImageFilter(m_dt_init, m_maskImage);

	TensorImageType::IndexType testIdx;
	testIdx[0]=11; testIdx[1]=22; testIdx[2]=0;



	ScalarImageIterator itGradMag(gradMagTensorImage, gradMagTensorImage->GetLargestPossibleRegion());
	ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
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

	ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
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
	ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
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

JointTensorEstimation::TensorImageType::Pointer JointTensorEstimation::UpdateTerms()
{
    TensorImageType::IndexType testIdx;
    testIdx[0]=49; testIdx[1]=21; testIdx[2] = 9;

    TensorUtilities utilsTens;
    CopyImage cpImage;

    TensorImageType::Pointer logTensorImage_n_1 = TensorImageType::New();
    cpImage.CopyTensorImage(m_dt_init,logTensorImage_n_1 );


    TotalEnergy totEnergy;
    totEnergy.ReadB0Image(m_B0Image);
    totEnergy.ReadBValue(1);
    totEnergy.ReadGradientImageList(m_GradList);
    totEnergy.ReadImageList(m_DWIList);
    totEnergy.ReadKappa(m_kappa);
    totEnergy.ReadSigma(m_Sigma);
    totEnergy.ReadMaskImage(m_maskImage);


    TensorImageType::Pointer logTensorImage_n = utilsTens.LogTensorImageFilter(m_dt_init,m_maskImage);

    TensorImageType::IndexType testIndex;
    testIndex[0]=54; testIndex[1]=21; testIndex[2]=1;



    ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());

    std::vector<RealType> Energy_vec;

//    std::cout << "In the Update Terms " << logTensorImage_n->GetPixel(testIndex) << std::endl;


    RealType gaussian = totEnergy.GaussianNoise(logTensorImage_n);
    RealType RegEnergy = totEnergy.RegularizationEnergy(logTensorImage_n);
    RealType totalEnergy =  gaussian + m_Lambda*RegEnergy;

     Energy_vec.push_back(totalEnergy);


    for (int i=0; i < m_Iterations; i++ )
    {




//    TensorImageType::Pointer delSim = ComputeDelSim_nonFrac(logTensorImage_n);
    TensorImageType::Pointer delSim = ComputeDelSim_Frac(logTensorImage_n);


    ScalarImageType::Pointer gradMagTensorImage = GradientLogMagTensorImage(logTensorImage_n);

    ScalarImageType::Pointer psiImage = ComputePsiImage(gradMagTensorImage);

    TensorImageType::Pointer LaplaceTensorImage = ComputeLaplaceTensor(logTensorImage_n);
    TensorImageType::Pointer FirstTerm = ComputeFirstTermDelReg(psiImage, LaplaceTensorImage);
    TensorImageType::Pointer SecTerm = ComputeSecondTermDelReg(psiImage, logTensorImage_n);
    TensorImageType::Pointer delRegTerm = ComputeDelReg(FirstTerm, SecTerm);

    TensorImageIterator itDelSim(delSim, delSim->GetLargestPossibleRegion());
    TensorImageIterator itDelReg(delRegTerm, delRegTerm->GetLargestPossibleRegion());
    TensorImageIterator itL_n(logTensorImage_n, logTensorImage_n->GetLargestPossibleRegion());
    TensorImageIterator itL_n_1(logTensorImage_n_1, logTensorImage_n_1->GetLargestPossibleRegion());

    for (itMask.GoToBegin(), itDelSim.GoToBegin(), itDelReg.GoToBegin(), itL_n.GoToBegin(), itL_n_1.GoToBegin();
    		!itMask.IsAtEnd(), !itDelReg.IsAtEnd(), !itDelSim.IsAtEnd(), !itL_n.IsAtEnd(), !itL_n_1.IsAtEnd();
    		++itMask, ++itDelReg, ++itDelSim, ++itL_n, ++itL_n_1)
    {
    	if (itMask.Get() != 0)
    	{
    		DiffusionTensorType L_temp;
    		L_temp = itL_n.Get() - (itDelSim.Get() + itDelReg.Get()*m_Lambda)*m_Step_size;
    		itL_n_1.Set(L_temp);

//    		std::cout << "L_n " << itL_n.Get() << std::endl;
//    		std::cout << "DelSim "  << itDelSim.Get() << std::endl;
//    		std::cout << "DelReg "  << itDelReg.Get() << std::endl;


    	}
    }

    logTensorImage_n = logTensorImage_n_1;
    RealType gaussian1 = totEnergy.GaussianNoise(logTensorImage_n_1);
    RealType RegEnergy1 = totEnergy.RegularizationEnergy(logTensorImage_n_1);
    RealType totalEnergy1 =  gaussian1 + m_Lambda*RegEnergy1;

    std::cout << "Energy " << totalEnergy1 << std::endl;
    std::cout << "Energy back " << Energy_vec.back() << std::endl;
    std::cout << "Data fidelity Energy " << gaussian1 << std::endl;
    std::cout << "Regularization Energy " << RegEnergy1 << std::endl;


    if (totalEnergy1 < Energy_vec.back())
    {
        logTensorImage_n = logTensorImage_n_1;

    	Energy_vec.push_back(totalEnergy);
    }
    else
    {
    	break;
    }


    }
    return logTensorImage_n;
}
