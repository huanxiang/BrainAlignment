/*==========================================================================================
 *
 *	Program:	TOMI
 *	Module:		BrainAlignment
 *	Language: C++
 *	Data:			$Date: 2014-03-17 10:47:53$
 *	Version:	$Revision: 0.1$
 *
 *	Copyright Philips Research China. All rights reserved.
 *
 *  This program aligns the brain image into the mni atlas space using affine registration.
 *
 *=========================================================================================*/


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAffineTransform.h"
#include "itkVersorRigid3DTransform.h"

#include "BrainAlignmentFilter.h"

int main(int argc, char* argv[])
{
	if(argc < 4)
	{
		std::cout<<"Usage: "<<argv[0]<<" subjectImage atlasPath downSampleFactor"<<std::endl;
		return EXIT_FAILURE;
	}

	// Typical type definitions
	typedef short		PixelType;
	const unsigned int Dimension = 3;
	typedef itk::Image<PixelType, Dimension>	ImageType;
	typedef itk::ImageFileReader<ImageType>		ReaderType;
	typedef itk::ImageFileWriter<ImageType>		WriterType;
	typedef BrainAlignment<ImageType>					AlignFilterType;
	typedef itk::VersorRigid3DTransform<double>				RigidTransformType;
	typedef itk::AffineTransform<double>							AffineTransformType;
	typedef itk::Image<float, Dimension>							FloatImageType;
	typedef itk::ImageFileWriter<FloatImageType>			FloatWriterType;

	// Load input image
	ReaderType::Pointer inputReader = ReaderType::New();
	inputReader->SetFileName(argv[1]);
	try
	{
		inputReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<<err<<std::endl;
		return EXIT_FAILURE;
	}

	ImageType::Pointer inputImage = inputReader->GetOutput();
	inputImage->DisconnectPipeline();


	// Align the input image and perform template matching with mni atlas
	AlignFilterType * filter = new AlignFilterType;
	filter->SetAtlasPath(argv[2]);
	filter->SetInput(inputImage);
	filter->SetNumberOfSamples(100000);
	filter->SetDownsampleFactor(::atof(argv[3]));
	filter->UpdateRSS();

	ImageType::Pointer rigidAlignedImage = filter->GetRigidOutput();

	// write rigid output
	typedef itk::ImageFileWriter<ImageType>	WriterType;
	WriterType::Pointer rigidWriter = WriterType::New();
	rigidWriter->SetInput(rigidAlignedImage);
	rigidWriter->SetFileName("rigidAlignedImage.nii");
	try
	{
		rigidWriter->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err << std::endl;
	}


	ImageType::Pointer affineAtlas = filter->GetAffineOutput();

	WriterType::Pointer affineWriter = WriterType::New();
	affineWriter->SetInput(affineAtlas);
	affineWriter->SetFileName("affineAlignedAtlas.nii");
	try
	{
		affineWriter->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err << std::endl;
	}

	ImageType::Pointer affineMask = filter->GetAffineMask();

	affineWriter->SetInput(affineMask);
	affineWriter->SetFileName("affineAlignedMask.nii");
	try
	{
		affineWriter->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err << std::endl;
	}

	FloatImageType::Pointer affineCSF = filter->GetAffineCSF();

	FloatWriterType::Pointer floatWriter = FloatWriterType::New();
	floatWriter->SetInput(affineCSF);
	floatWriter->SetFileName("affineAlignedCSF.nii");
	try
	{
		floatWriter->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err << std::endl;
	}

	FloatImageType::Pointer affineWM = filter->GetAffineWM();

	floatWriter->SetInput(affineWM);
	floatWriter->SetFileName("affineAlignedWM.nii");
	try
	{
		floatWriter->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err << std::endl;
	}

	FloatImageType::Pointer affineGM = filter->GetAffineGM();

	floatWriter->SetInput(affineGM);
	floatWriter->SetFileName("affineAlignedGM.nii");
	try
	{
		floatWriter->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err << std::endl;
	}

	ImageType::Pointer affineLabel = filter->GetAffineLabel();

	affineWriter->SetInput(affineLabel);
	affineWriter->SetFileName("affineAlignedLabel.nii");
	try
	{
		affineWriter->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err << std::endl;
	}

	return EXIT_SUCCESS;
}
