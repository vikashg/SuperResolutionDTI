#include "WeightedLeastSquares_new.h"
#include <math.h>
using namespace std;


void WeightedLeastSquares::ReadDWIListLR(ImageListType listDWI)
{
  m_DWIListLR = listDWI;
}

void WeightedLeastSquares::ReadDWIListHR(ImageListType listDWI)
{
  m_DWIListHR = listDWI;
}

void WeightedLeastSquares::ReadTensorImage(TensorImageType::Pointer tensorImage)
{
	m_tensorImage_init = tensorImage;
}

void WeightedLeastSquares::ReadHRMask(ScalarImageType::Pointer maskImage)
{
      m_HRmask = maskImage;
}

void WeightedLeastSquares::ReadLRMask(ScalarImageType::Pointer maskImage)
{
	m_LRmask = maskImage;	
}

void WeightedLeastSquares::ReadBVal(RealType bval)
{
	m_BVal = bval;
}

void WeightedLeastSquares::ReadB0ImageHR(ScalarImageType::Pointer B0Image)
{
	m_B0Image_HR = B0Image;
}

void WeightedLeastSquares::ReadB0ImageLR(ScalarImageType::Pointer B0Image)
{
	m_B0Image_LR = B0Image;
}

void WeightedLeastSquares::ReadMapMatrixLR2HR(SparseMatrixType map)
{
	m_MapLR2HR = map;
}

void WeightedLeastSquares::ReadGradientList(VectorImageListType gradList)
{
	m_GradList = gradList;
}

WeightedLeastSquares::ImageListType WeightedLeastSquares::ComputePredictedImage(TensorImageType::Pointer tensorImage)
{

//Compute Predicted Imge
	CopyImage cpImage;
	TensorUtilities utilsTensor;
	
	int numOfImages = m_DWIListHR.size();
// Evaluate W

	
	ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	ScalarImageIterator itB0(m_B0Image_HR, m_B0Image_HR->GetLargestPossibleRegion());
	TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());

	ImageListType PredImageList;

	for (int i=0; i < numOfImages; i++)
	{
		ScalarImageType::Pointer predImage_HR_i = ScalarImageType::New();
		cpImage.CopyScalarImage(m_B0Image_HR, predImage_HR_i);
		ScalarImageIterator itPredHR(predImage_HR_i, predImage_HR_i->GetLargestPossibleRegion());
		

		for (itHRMask.GoToBegin(), itPredHR.GoToBegin(), itB0.GoToBegin(), itTens.GoToBegin(); 
		     !itPredHR.IsAtEnd(), !itHRMask.IsAtEnd(), !itB0.IsAtEnd(), !itTens.IsAtEnd();
		    ++itPredHR, ++itHRMask, ++itB0, ++itTens)
		{
		  	
		if (itHRMask.Get() != 0)
			{
			vnl_vector<double> g_i_temp = m_GradList[i]->GetPixel(itHRMask.GetIndex()).GetVnlVector();
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

			RealType atten_i = exp(temp(0,0)*(-1)*m_BVal);
			
			itPredHR.Set(atten_i);
			}

		}

		PredImageList.push_back(predImage_HR_i);

	}
// Compute the difference
	return PredImageList;
}



vnl_matrix<double> WeightedLeastSquares::ComputeJacobian(DiffusionTensorType D)
{
	TensorUtilities utils;
	
//	std::cout << "Jacobian Computation start" << D << std::endl;
	vnl_matrix<double> P_upper = utils.CholeskyDecomposition(D);

//	std::cout << "P_upper " << P_upper << std::endl;	

	vnl_vector<double> Rho_vec;
	Rho_vec.set_size(7);
	Rho_vec[0] = 1;
	Rho_vec[1]=  P_upper(0,0);  // rho 2
	Rho_vec[2] = P_upper(1,1); // rho 3
	Rho_vec[3] = P_upper(2,2); // rho 4
	Rho_vec[4] = P_upper(0,1); // rho 5
	Rho_vec[5] = P_upper(1,2); // rho 6
	Rho_vec[6] = P_upper(0,2); // rho 7


	vnl_matrix<double> J; 
	J.set_size(7,7); J.fill(0.0);

	J(0,0)=1; 
	
	J(1,1) = 2*Rho_vec.get(1);
	J(4,1) = Rho_vec.get(4);
	J(6,1) = Rho_vec.get(6);

	J(2,2) = 2*Rho_vec.get(2);
	J(5,2) = Rho_vec.get(5);

	J(3,3) = 2*Rho_vec.get(3);

	J(2,4) = 2*Rho_vec.get(4);
	J(4,4) = Rho_vec.get(1);

	J(3,5) = 2*Rho_vec.get(5);
	J(5,5) =  Rho_vec.get(2);

	J(3,6) = 2*Rho_vec.get(6);
	J(6,6) = Rho_vec.get(1);

	return J;	
}

WeightedLeastSquares::VnlVectorType WeightedLeastSquares::ComputeWeightMatrixRow(VectorType G)
{
	RealType Gx, Gy, Gz;
	Gx=G[0]; Gy=G[1]; Gz=G[2];

	vnl_vector<double> W_i;
	W_i.set_size(7);
	W_i(0) = -m_BVal*Gx*Gx;
	W_i(1) = -m_BVal*Gy*Gy;
	W_i(2) = -m_BVal*Gz*Gz;
	W_i(3) = -2*m_BVal*Gx*Gy;
	W_i(4) = -2*m_BVal*Gy*Gz;
	W_i(5) = -2*m_BVal*Gx*Gz;

	return W_i;	
}

WeightedLeastSquares::ImageListType WeightedLeastSquares::ComputeDifferenceImage(ImageListType PredImageList)
{

	ImageListType DifferenceImageList;
	
	int numOfImages = m_DWIListHR.size();

	ScalarImageIterator itB0(m_B0Image_HR, m_B0Image_HR->GetLargestPossibleRegion());
	CopyImage cpImage;
	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());	

	for (int i=0; i < numOfImages; i++)
	{
		ScalarImageType::Pointer diffImage_i = ScalarImageType::New();
		cpImage.CopyScalarImage(m_B0Image_HR, diffImage_i);	
		ScalarImageIterator itObs_i(m_DWIListHR[i], m_DWIListHR[i]->GetLargestPossibleRegion());
		ScalarImageIterator itDiff(diffImage_i, diffImage_i->GetLargestPossibleRegion());
		ScalarImageIterator itPred(PredImageList[i], PredImageList[i]->GetLargestPossibleRegion());


		for (itMask.GoToBegin(), itMask.GoToBegin(), itDiff.GoToBegin(), itObs_i.GoToBegin(), itB0.GoToBegin(), itPred.GoToBegin();
			!itMask.IsAtEnd(), !itDiff.IsAtEnd(), !itObs_i.IsAtEnd(), !itB0.IsAtEnd(), !itPred.IsAtEnd();
			++itMask, ++itDiff, ++itB0, ++itObs_i, ++itPred)
		if (itMask.Get() !=0)
		{
			double diffTemp = log(itObs_i.Get()/itB0.Get()) - itPred.Get();
			itDiff.Set(diffTemp);

		}
		DifferenceImageList.push_back(diffImage_i);	
		
	}	
		
	return DifferenceImageList;
}

WeightedLeastSquares::VectorialImageTensorType::Pointer  WeightedLeastSquares::ComputeDelSim(TensorImageType::Pointer tensorImage)
{
	ScalarImageIterator itHRMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	
	TensorImageType::Pointer delSim = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(tensorImage, delSim);
	
	TensorImageIterator itDelSim(delSim, delSim->GetLargestPossibleRegion());

	int numOfImages = m_DWIListHR.size();

	//ComputePredictedImage;
	ImageListType PredImageList = ComputePredictedImage(tensorImage);

	//ComputeDifferenceImageList
	ImageListType DiffImageList = ComputeDifferenceImage(PredImageList);

	ScalarImageType::IndexType tempIndex;
	tempIndex[0] = 70; 
	tempIndex[1] = 109;
	tempIndex[2] = 81;

	VectorialImageTensorType::Pointer DelSim = VectorialImageTensorType::New();
	DelSim->SetOrigin(m_HRmask->GetOrigin());
	DelSim->SetDirection(m_HRmask->GetDirection());
	DelSim->SetSpacing(m_HRmask->GetSpacing());
	DelSim->SetRegions(m_HRmask->GetLargestPossibleRegion());
	DelSim->Allocate();
	
	VectorialTensorType  ZeroVecTensor; ZeroVecTensor.Fill(0);
	DelSim->FillBuffer(ZeroVecTensor);	
	
	for(itHRMask.GoToBegin(), itDelSim.GoToBegin(); !itHRMask.IsAtEnd(), !itDelSim.IsAtEnd() ; ++itHRMask, ++itDelSim)
	{
	
	 if (itHRMask.Get() != 0)
	{
		vnl_matrix<double> W; W.set_size(numOfImages, 6);
		vnl_vector<double> S_vec; S_vec.set_size(numOfImages);
		vnl_diag_matrix<double> S;
		vnl_vector<double> r;
		r.set_size(numOfImages);	
		DiffusionTensorType D = tensorImage->GetPixel(itHRMask.GetIndex());
	//	vnl_matrix<double> J = ComputeJacobian(D);		

		 for(int i=0; i < numOfImages; i++)
		{
			
			//Create W 
			vnl_vector<double> W_row_i = ComputeWeightMatrixRow(m_GradList[i]->GetPixel(itHRMask.GetIndex()));
			W.set_row(i, W_row_i);		
		
			// Create S
			S_vec[i] = PredImageList[i]->GetPixel(itHRMask.GetIndex());
			//Create r
			r[i] = DiffImageList[i]->GetPixel(itHRMask.GetIndex());	
				
		}
			vnl_vector<double> delF; 
			S.set(S_vec);
			delF = W.transpose()*S*r;
			VectorialTensorType tempDelF;
			tempDelF.SetVnlVector(delF);
			DelSim->SetPixel(itHRMask.GetIndex(), tempDelF);
	
	}
	}
	
	return DelSim;
}


WeightedLeastSquares::VectorialImageTensorType::Pointer WeightedLeastSquares::MakeGammaImage(TensorImageType::Pointer tensorImage)
{

	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	TensorImageIterator itTens(tensorImage, tensorImage->GetLargestPossibleRegion());

	VectorialImageTensorType::Pointer GammaImage = VectorialImageTensorType::New();
	GammaImage->SetOrigin(m_HRmask->GetOrigin());
	GammaImage->SetDirection(m_HRmask->GetDirection());
	GammaImage->SetSpacing(m_HRmask->GetSpacing());
	GammaImage->SetRegions(m_HRmask->GetLargestPossibleRegion());
	GammaImage->Allocate();
	

	VectorialTensorType  ZeroVecTensor; ZeroVecTensor.Fill(0);
	GammaImage->FillBuffer(ZeroVecTensor);



	for (itMask.GoToBegin(), itTens.GoToBegin(); !itMask.IsAtEnd(), !itTens.IsAtEnd(); ++itMask, ++itTens)
	{
	if (itMask.Get() != 0)
	{
		VectorialTensorType temp;
		DiffusionTensorType D = itTens.Get();
//		temp[0] = log(m_B0Image_HR->GetPixel(itMask.GetIndex()));
		temp[0] = D(0,0);
		temp[1] = D(1,1);
		temp[2] = D(2,2);
		temp[3] = D(0,1);
		temp[4] = D(1,2);
		temp[5] = D(0,2);	
		
		GammaImage->SetPixel(itMask.GetIndex(), temp);
	}
		
	}
	
	return GammaImage;
}



WeightedLeastSquares::TensorImageType::Pointer WeightedLeastSquares::ConvertGammaVector2DT(VectorialImageTensorType::Pointer vecTensorImage)
{
	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	VecTensorImageIterator itVec(vecTensorImage , vecTensorImage->GetLargestPossibleRegion());

	TensorImageType::Pointer tempTensorImage = TensorImageType::New();
	CopyImage  cpImage;
	cpImage.CopyTensorImage(m_tensorImage_init, tempTensorImage);
	
	TensorImageIterator itTensor(tempTensorImage, tempTensorImage->GetLargestPossibleRegion());
	
	for (itMask.GoToBegin(), itTensor.GoToBegin() ; !itMask.IsAtEnd(), !itTensor.IsAtEnd(); 
		++itMask, ++itTensor)
	{
	  if (itMask.Get() != 0)
	  {
		VectorialTensorType gamma;
	   	gamma  = vecTensorImage->GetPixel(itMask.GetIndex());
		
		DiffusionTensorType D;
/*		D(0,0) = gamma[1];
		D(1,1) = gamma[2];
		D(2,2) = gamma[3];
		D(0,1) = gamma[4];
		D(1,2) = gamma[5];
		D(0,2) = gamma[6]; 
*/		
		D(0,0) = gamma[0];
		D(1,1) = gamma[1];
		D(2,2) = gamma[2];
		D(0,1) = gamma[3];
		D(1,2) = gamma[4];
		D(0,2) = gamma[5];
 
		itTensor.Set(D);		     
	  }
	}

	return tempTensorImage;
}



WeightedLeastSquares::VectorialImageTensorType::Pointer WeightedLeastSquares::ConvertDT2Vector(TensorImageType::Pointer tensorImage)
{
	
	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	TensorImageIterator itTensor(tensorImage, tensorImage->GetLargestPossibleRegion());

	VectorialImageTensorType::Pointer vecImageTensor = VectorialImageTensorType::New();
        vecImageTensor->SetOrigin(m_HRmask->GetOrigin());
        vecImageTensor->SetDirection(m_HRmask->GetDirection());
        vecImageTensor->SetSpacing(m_HRmask->GetSpacing());
        vecImageTensor->SetRegions(m_HRmask->GetLargestPossibleRegion());
        vecImageTensor->Allocate();

        VectorialTensorType  ZeroVecTensor; ZeroVecTensor.Fill(0);

	vecImageTensor->FillBuffer(ZeroVecTensor);

	TensorUtilities utils;	

	for (itMask.GoToBegin(), itTensor.GoToBegin(); !itMask.IsAtEnd(); ++itTensor, ++itMask)
	{
		if (itMask.Get() != 0)
		{
		  DiffusionTensorType D = itTensor.Get();
		  vnl_matrix<double> P = utils.CholeskyDecomposition(D);

	   vnl_vector<double> Rho_vec;
	    Rho_vec.set_size(7);
	    Rho_vec[0] = log(m_B0Image_HR->GetPixel(itMask.GetIndex()));
	 /*   Rho_vec[1] = P(0,0);  // rho 2
	    Rho_vec[2] = P(1,1); // rho 3
	    Rho_vec[3] = P(2,2); // rho 4
	    Rho_vec[4] = P(0,1); // rho 5
	    Rho_vec[5] = P(1,2); // rho 6
	    Rho_vec[6] = P(0,2); // rho 7		 
	*/

             Rho_vec[1] = D(0,0);
	     Rho_vec[2] = D(1,1);
	     Rho_vec[3] = D(2,2);
		Rho_vec[4] = D(0,1);
		Rho_vec[5] = D(1,2);
		Rho_vec[6] = D(0,2);	    
	    VectorialTensorType tempVec; tempVec.SetVnlVector(Rho_vec);	
		
	    vecImageTensor->SetPixel(itMask.GetIndex(), tempVec);
						
		}
	}
	
	return vecImageTensor;
}




WeightedLeastSquares::RealType WeightedLeastSquares::NLSEnergy(TensorImageType::Pointer tensorImage)
{
	ImageListType PredImageList = ComputePredictedImage(tensorImage);
	ImageListType DiffImageList = ComputeDifferenceImage(PredImageList);

	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());
	int numOfImage = m_DWIListHR.size();
	
	RealType Sum =0;
	
	for (int i=0; i < numOfImage; i++)
	{
		ScalarImageIterator itDiff(DiffImageList[i], DiffImageList[i]->GetLargestPossibleRegion());
		for (itMask.GoToBegin(), itDiff.GoToBegin() ; !itMask.IsAtEnd(), !itDiff.IsAtEnd(); ++itMask, ++itDiff)
		{
		  if (itMask.Get() != 0)
		  {
			RealType tempNoise = itDiff.Get();
			Sum = Sum + tempNoise*tempNoise;	
		   }
		}
	}

	return Sum;	
}


void WeightedLeastSquares::UpdateTerms()
{


	VectorialImageTensorType::Pointer vecTensorImage_n = MakeGammaImage(m_tensorImage_init);
	VectorialImageTensorType::Pointer vecTensorImage_n_1 = VectorialImageTensorType::New();

	
	vecTensorImage_n_1->SetOrigin(vecTensorImage_n->GetOrigin());
	vecTensorImage_n_1->SetDirection(vecTensorImage_n->GetDirection());
	vecTensorImage_n_1->SetSpacing(vecTensorImage_n->GetSpacing());
	vecTensorImage_n_1->SetRegions(vecTensorImage_n->GetLargestPossibleRegion());
	vecTensorImage_n_1->Allocate();
        
	VectorialTensorType  ZeroVecTensor; ZeroVecTensor.Fill(0);
        vecTensorImage_n_1->FillBuffer(ZeroVecTensor);

	ScalarImageIterator itMask(m_HRmask, m_HRmask->GetLargestPossibleRegion());

	RealType Energy =0;
	Energy = NLSEnergy(m_tensorImage_init); 

	std::vector<RealType> Energy_vec;
	Energy_vec.push_back(Energy);

	std::cout << "Energy " << Energy << std::endl;

	int numOfIterations =5;
	double dt =0.05;

	TensorImageType::Pointer tensorImage_n = m_tensorImage_init;

  ScalarImageType::IndexType tempIndex;
         tempIndex[0] = 70;
         tempIndex[1] = 109;
         tempIndex[2] = 81;

	for (int i=0; i < numOfIterations; i++)
	{	
		
		VectorialImageTensorType::Pointer DelSim_n = ComputeDelSim(tensorImage_n);
		VecTensorImageIterator itVec_n(vecTensorImage_n, vecTensorImage_n->GetLargestPossibleRegion());
		VecTensorImageIterator itDelSim(DelSim_n , DelSim_n->GetLargestPossibleRegion());
		VecTensorImageIterator itVec_n_1(vecTensorImage_n_1, vecTensorImage_n_1->GetLargestPossibleRegion()); 	


		for (itMask.GoToBegin(), itVec_n.GoToBegin(), itVec_n_1.GoToBegin(), itDelSim.GoToBegin(); 
			!itMask.IsAtEnd(), !itVec_n.IsAtEnd(), !itVec_n_1.IsAtEnd(), !itDelSim.IsAtEnd();
			++itMask, ++itVec_n, ++itVec_n_1, ++itDelSim)
		{
			if (itMask.Get() != 0)
			{

			VectorialTensorType tempVector;
			tempVector = itVec_n.Get() + itDelSim.Get()*dt;		
			itVec_n_1.Set(tempVector);
			
//				std::cout << itDelSim.Get() << std::endl;
//				std::cout << itVec_n.Get() << std::endl;
//				std::cout << tempVector << std::endl;
					
//				std::cout << " " << std::endl;
			}

		} 	  			
		
		TensorImageType::Pointer tensorImage_n_1 = ConvertGammaVector2DT(vecTensorImage_n_1);
		
		RealType Energy_n_1 = NLSEnergy(tensorImage_n_1);

		std::cout << "Energy_n_1 " << Energy_n_1 << std::endl;
		
		if ( Energy_n_1 < Energy_vec.back() )
		{
		vecTensorImage_n = vecTensorImage_n_1;
		Energy_vec.push_back(Energy_n_1);	
		tensorImage_n = tensorImage_n_1;			

		}
		else
		{
		 break;
		}
		
	}

	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;
	TensorWriterType::Pointer tensorWriter = TensorWriterType::New();
	tensorWriter->SetFileName("EstimatedTensors.nii.gz");
	tensorWriter->SetInput(tensorImage_n);
	tensorWriter->Update();

}
