using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Services;
using System.IO;
using weka.classifiers.bayes.net;
using weka.classifiers;
using java.io;
using weka.core;
using weka.filters;
using weka.core.converters;

namespace InferenceWebService
{
    /// <summary>
    /// Summary description for Service1
    /// </summary>
    [WebService(Namespace = "http://tempuri.org/")]
    [WebServiceBinding(ConformsTo = WsiProfiles.BasicProfile1_1)]
    [System.ComponentModel.ToolboxItem(false)]
    // To allow this Web Service to be called from script, using ASP.NET AJAX, uncomment the following line. 
    // [System.Web.Script.Services.ScriptService]
    public class Service1 : System.Web.Services.WebService
    {

        
        [WebMethod]
        public bool DoInference(InputData InputDataValue, ref string strResult)
        {
            //载入模型
            Classifier cfs = null;//分类器
            cfs = (weka.classifiers.Classifier)LoadModel(".\\InferenceWebService\\model\\BayesNet.model");

            // Declare two numeric attributes
             weka.core.Attribute Attribute1 = new weka.core.Attribute("timeorientation");
             weka.core.Attribute Attribute2 = new weka.core.Attribute("placeorientation");
             weka.core.Attribute Attribute3 = new weka.core.Attribute("Languageimmediaterecall");
             weka.core.Attribute Attribute4 = new weka.core.Attribute("Attentionandcalculation");
             weka.core.Attribute Attribute5 = new weka.core.Attribute("shortmemory");
             weka.core.Attribute Attribute6 = new weka.core.Attribute("namingobjects");
             weka.core.Attribute Attribute7 = new weka.core.Attribute("languageretell");
             weka.core.Attribute Attribute8 = new weka.core.Attribute("readingcomprehension");
             weka.core.Attribute Attribute9 = new weka.core.Attribute("languageunderstanding");
             weka.core.Attribute Attribute10 = new weka.core.Attribute("languageexpression");
             weka.core.Attribute Attribute11 = new weka.core.Attribute("drawgraph");
             weka.core.Attribute Attribute12 = new weka.core.Attribute("Visualspaceandexecutiveability");
             weka.core.Attribute Attribute13 = new weka.core.Attribute("naming");
             weka.core.Attribute Attribute14 = new weka.core.Attribute("memory");
             weka.core.Attribute Attribute15 = new weka.core.Attribute("attention");
             weka.core.Attribute Attribute16 = new weka.core.Attribute("language");
             weka.core.Attribute Attribute17 = new weka.core.Attribute("animalnumber");
             weka.core.Attribute Attribute18 = new weka.core.Attribute("abstractability");
             weka.core.Attribute Attribute19 = new weka.core.Attribute("MoCadelayrecall");
             weka.core.Attribute Attribute20 = new weka.core.Attribute("orientaion");
             weka.core.Attribute Attribute21 = new weka.core.Attribute("PhysicalSelf-maintenance");
             weka.core.Attribute Attribute22 = new weka.core.Attribute("grippingability");
             weka.core.Attribute Attribute23 = new weka.core.Attribute("word1");
             weka.core.Attribute Attribute24 = new weka.core.Attribute("word2");
             weka.core.Attribute Attribute25 = new weka.core.Attribute("word3");
             weka.core.Attribute Attribute26 = new weka.core.Attribute("wordaverage");
             weka.core.Attribute Attribute27 = new weka.core.Attribute("worddelayrecall");
             weka.core.Attribute Attribute28 = new weka.core.Attribute("originalwordrecognition");
             weka.core.Attribute Attribute29 = new weka.core.Attribute("Newwordrecognize");
             weka.core.Attribute Attribute30 = new weka.core.Attribute("graphcopy");
             weka.core.Attribute Attribute31 = new weka.core.Attribute("graphimmediaterecall");
             weka.core.Attribute Attribute32 = new weka.core.Attribute("graphdelayrecall");
             weka.core.Attribute Attribute33 = new weka.core.Attribute("lineA");
             weka.core.Attribute Attribute34 = new weka.core.Attribute("lineB");
             weka.core.Attribute Attribute35 = new weka.core.Attribute("GDS");
             weka.core.Attribute Attribute36 = new weka.core.Attribute("CDR");
             
             
             // Declare the class attribute along with its values
             FastVector fvClassVal = new FastVector(3);
             fvClassVal.addElement("Normal");
             fvClassVal.addElement("MCI");
             fvClassVal.addElement("AD");
             weka.core.Attribute ClassAttribute = new weka.core.Attribute("Result", fvClassVal);
             
             // Declare the feature vector
             FastVector fvWekaAttributes = new FastVector(37);
             fvWekaAttributes.addElement(Attribute1);    
             fvWekaAttributes.addElement(Attribute2);    
             fvWekaAttributes.addElement(Attribute3);
             fvWekaAttributes.addElement(Attribute4);
             fvWekaAttributes.addElement(Attribute5);
             fvWekaAttributes.addElement(Attribute6);
             fvWekaAttributes.addElement(Attribute7);
             fvWekaAttributes.addElement(Attribute8);
             fvWekaAttributes.addElement(Attribute9);
             fvWekaAttributes.addElement(Attribute10);
             fvWekaAttributes.addElement(Attribute11);
             fvWekaAttributes.addElement(Attribute12);
             fvWekaAttributes.addElement(Attribute13);
             fvWekaAttributes.addElement(Attribute14);
             fvWekaAttributes.addElement(Attribute15);
             fvWekaAttributes.addElement(Attribute16);
             fvWekaAttributes.addElement(Attribute17);
             fvWekaAttributes.addElement(Attribute18);
             fvWekaAttributes.addElement(Attribute19);
             fvWekaAttributes.addElement(Attribute20);
             fvWekaAttributes.addElement(Attribute21);
             fvWekaAttributes.addElement(Attribute22);
             fvWekaAttributes.addElement(Attribute23);
             fvWekaAttributes.addElement(Attribute24);
             fvWekaAttributes.addElement(Attribute25);
             fvWekaAttributes.addElement(Attribute26);
             fvWekaAttributes.addElement(Attribute27);
             fvWekaAttributes.addElement(Attribute28);
             fvWekaAttributes.addElement(Attribute29);
             fvWekaAttributes.addElement(Attribute30);
             fvWekaAttributes.addElement(Attribute31);
             fvWekaAttributes.addElement(Attribute32);
             fvWekaAttributes.addElement(Attribute33);
             fvWekaAttributes.addElement(Attribute34);
             fvWekaAttributes.addElement(Attribute35);
             fvWekaAttributes.addElement(Attribute36);
             fvWekaAttributes.addElement(ClassAttribute);

             // Create an empty training set
             Instances isTrainingSet = new Instances("ADData", fvWekaAttributes, 37);//-----------------
             // Set class index
             isTrainingSet.setClassIndex(36);

             double[] vals;
             vals = new double[37];
             vals[0] = InputDataValue.timeorientation;
             vals[1] = InputDataValue.placeorientation;
             vals[2] = InputDataValue.Languageimmediaterecall;
             vals[3] = InputDataValue.Attentionandcalculation;
             vals[4] = InputDataValue.shortmemory;
             vals[5] = InputDataValue.namingobjects;
             vals[6] = InputDataValue.languageretell;
             vals[7] = InputDataValue.readingcomprehension;
             vals[8] = InputDataValue.languageunderstanding;
             vals[9] = InputDataValue.languageexpression;
             vals[10] = InputDataValue.drawgraph;
             vals[11] = InputDataValue.Visualspaceandexecutiveability;
             vals[12] = InputDataValue.naming;
             vals[13] = InputDataValue.memory;
             vals[14] = InputDataValue.attention;
             vals[15] = InputDataValue.language;
             vals[16] = InputDataValue.animalnumber;
             vals[17] = InputDataValue.abstractability;
             vals[18] = InputDataValue.MoCadelayrecall;
             vals[19] = InputDataValue.orientaion;
             vals[20] = InputDataValue.PhysicalSelfmaintenance;
             vals[21] = InputDataValue.grippingability;
             vals[22] = InputDataValue.word1;
             vals[23] = InputDataValue.word2;
             vals[24] = InputDataValue.word3;
             vals[25] = InputDataValue.wordaverage;
             vals[26] = InputDataValue.worddelayrecall;
             vals[27] = InputDataValue.originalwordrecognition;
             vals[28] = InputDataValue.Newwordrecognize;
             vals[29] = InputDataValue.graphcopy;
             vals[30] = InputDataValue.graphimmediaterecall;
             vals[31] = InputDataValue.graphdelayrecall;
             vals[32] = InputDataValue.lineA;
             vals[33] = InputDataValue.lineB;
             vals[34] = InputDataValue.GDS;
             vals[35] = InputDataValue.CDR;

             // Create the instance
             Instance iExample = new Instance(37);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(0), vals[0]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(1), vals[1]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(2), vals[2]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(3), vals[3]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(4), vals[4]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(5), vals[5]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(6), vals[6]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(7), vals[7]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(8), vals[8]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(9), vals[9]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(10), vals[10]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(11), vals[11]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(12), vals[12]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(13), vals[13]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(14), vals[14]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(15), vals[15]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(16), vals[16]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(17), vals[17]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(18), vals[18]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(19), vals[19]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(20), vals[20]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(21), vals[21]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(22), vals[22]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(23), vals[23]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(24), vals[24]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(25), vals[25]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(26), vals[26]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(27), vals[27]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(28), vals[28]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(29), vals[29]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(30), vals[30]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(31), vals[31]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(32), vals[32]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(33), vals[33]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(34), vals[34]);
             iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(35), vals[35]);
             //iExample.setValue((weka.core.Attribute)fvWekaAttributes.elementAt(36), "MCI");

             // add the instance
             isTrainingSet.add(iExample);

             iExample.setDataset(isTrainingSet);

             double[] fDistribution = cfs.distributionForInstance(iExample);
             double predict = cfs.classifyInstance(iExample);

        
            if(fDistribution[0]>=fDistribution[1] && fDistribution[0]>=fDistribution[2])
               strResult = "Normal";
            else if(fDistribution[1]>=fDistribution[0] && fDistribution[1]>=fDistribution[2])
               strResult = "MCI";
            else
               strResult = "AD";
             
         
             return true;

        }

        [WebMethod]
        public bool DoTraining(InputData InputDataValue)
        {

            //////////////////////////////////////////////////////////////////////////
            //1,将新的数据写入案例库 
            FastVector atts;
            atts = new FastVector();
            FastVector attVals;
            Instances data;
            double[] vals;

            if (System.IO.File.Exists(".\\InferenceWebService\\Casebase\\Casebase.arff"))
            {
                ArffLoader loader = new ArffLoader();
                loader.setFile(new java.io.File(".\\InferenceWebService\\Casebase\\Casebase.arff"));
                data = loader.getDataSet(); //把数据给instances

                vals = new double[data.numAttributes()];
                vals[0] = InputDataValue.timeorientation;
                vals[1] = InputDataValue.placeorientation;
                vals[2] = InputDataValue.Languageimmediaterecall;
                vals[3] = InputDataValue.Attentionandcalculation;
                vals[4] = InputDataValue.shortmemory;
                vals[5] = InputDataValue.namingobjects;
                vals[6] = InputDataValue.languageretell;
                vals[7] = InputDataValue.readingcomprehension;
                vals[8] = InputDataValue.languageunderstanding;
                vals[9] = InputDataValue.languageexpression;
                vals[10] = InputDataValue.drawgraph;
                vals[11] = InputDataValue.Visualspaceandexecutiveability;
                vals[12] = InputDataValue.naming;
                vals[13] = InputDataValue.memory;
                vals[14] = InputDataValue.attention;
                vals[15] = InputDataValue.language;
                vals[16] = InputDataValue.animalnumber;
                vals[17] = InputDataValue.abstractability;
                vals[18] = InputDataValue.MoCadelayrecall;
                vals[19] = InputDataValue.orientaion;
                vals[20] = InputDataValue.PhysicalSelfmaintenance;
                vals[21] = InputDataValue.grippingability;
                vals[22] = InputDataValue.word1;
                vals[23] = InputDataValue.word2;
                vals[24] = InputDataValue.word3;
                vals[25] = InputDataValue.wordaverage;
                vals[26] = InputDataValue.worddelayrecall;
                vals[27] = InputDataValue.originalwordrecognition;
                vals[28] = InputDataValue.Newwordrecognize;
                vals[29] = InputDataValue.graphcopy;
                vals[30] = InputDataValue.graphimmediaterecall;
                vals[31] = InputDataValue.graphdelayrecall;
                vals[32] = InputDataValue.lineA;
                vals[33] = InputDataValue.lineB;
                vals[34] = InputDataValue.GDS;
                vals[35] = InputDataValue.CDR;

                if (InputDataValue.strResult == "Normal")
                {
                    vals[36] = data.attribute(36).indexOfValue("Normal");//InputDataValue.Result;
                }
                else if (InputDataValue.strResult == "MCI")
                {
                    vals[36] = data.attribute(36).indexOfValue("MCI");//InputDataValue.Result;
                }
                else
                {
                    vals[36] = data.attribute(36).indexOfValue("AD");//InputDataValue.Result;
                }

                data.add(new Instance(1.0, vals));

                ArffSaver saver = new ArffSaver();
                saver.setInstances(data);
                saver.setFile(new java.io.File(".\\InferenceWebService\\Casebase\\Casebase.arff"));
                saver.writeBatch(); 

            }
            else
            {
               
                atts.addElement(new weka.core.Attribute("timeorientation"));
                atts.addElement(new weka.core.Attribute("placeorientation"));
                atts.addElement(new weka.core.Attribute("Languageimmediaterecall"));
                atts.addElement(new weka.core.Attribute("Attentionandcalculation"));
                atts.addElement(new weka.core.Attribute("shortmemory"));
                atts.addElement(new weka.core.Attribute("namingobjects"));
                atts.addElement(new weka.core.Attribute("languageretell"));
                atts.addElement(new weka.core.Attribute("readingcomprehension"));
                atts.addElement(new weka.core.Attribute("languageunderstanding"));
                atts.addElement(new weka.core.Attribute("languageexpression"));
                atts.addElement(new weka.core.Attribute("drawgraph"));
                atts.addElement(new weka.core.Attribute("Visualspaceandexecutiveability"));
                atts.addElement(new weka.core.Attribute("naming"));
                atts.addElement(new weka.core.Attribute("memory"));
                atts.addElement(new weka.core.Attribute("attention"));
                atts.addElement(new weka.core.Attribute("language"));
                atts.addElement(new weka.core.Attribute("animalnumber"));
                atts.addElement(new weka.core.Attribute("abstractability"));
                atts.addElement(new weka.core.Attribute("MoCadelayrecall"));
                atts.addElement(new weka.core.Attribute("orientaion"));
                atts.addElement(new weka.core.Attribute("PhysicalSelf-maintenance"));
                atts.addElement(new weka.core.Attribute("grippingability"));
                atts.addElement(new weka.core.Attribute("word1"));
                atts.addElement(new weka.core.Attribute("word2"));
                atts.addElement(new weka.core.Attribute("word3"));
                atts.addElement(new weka.core.Attribute("wordaverage"));
                atts.addElement(new weka.core.Attribute("worddelayrecall"));
                atts.addElement(new weka.core.Attribute("originalwordrecognition"));
                atts.addElement(new weka.core.Attribute("Newwordrecognize"));
                atts.addElement(new weka.core.Attribute("graphcopy"));
                atts.addElement(new weka.core.Attribute("graphimmediaterecall"));
                atts.addElement(new weka.core.Attribute("graphdelayrecall"));
                atts.addElement(new weka.core.Attribute("lineA"));
                atts.addElement(new weka.core.Attribute("lineB"));
                atts.addElement(new weka.core.Attribute("GDS"));
                atts.addElement(new weka.core.Attribute("CDR")); //36
                //atts.addElement(new weka.core.Attribute("Result"));
                data = new Instances("ADData", atts, 0);

                attVals = new FastVector();
                attVals.addElement("Normal");
                attVals.addElement("MCI");
                attVals.addElement("AD");
                atts.addElement(new weka.core.Attribute("Result", attVals));
                
                vals = new double[data.numAttributes()];
                vals[0] = InputDataValue.timeorientation;
                vals[1] = InputDataValue.placeorientation;
                vals[2] = InputDataValue.Languageimmediaterecall;
                vals[3] = InputDataValue.Attentionandcalculation;
                vals[4] = InputDataValue.shortmemory;
                vals[5] = InputDataValue.namingobjects;
                vals[6] = InputDataValue.languageretell;
                vals[7] = InputDataValue.readingcomprehension;
                vals[8] = InputDataValue.languageunderstanding;
                vals[9] = InputDataValue.languageexpression;
                vals[10] = InputDataValue.drawgraph;
                vals[11] = InputDataValue.Visualspaceandexecutiveability;
                vals[12] = InputDataValue.naming;
                vals[13] = InputDataValue.memory;
                vals[14] = InputDataValue.attention;
                vals[15] = InputDataValue.language;
                vals[16] = InputDataValue.animalnumber;
                vals[17] = InputDataValue.abstractability;
                vals[18] = InputDataValue.MoCadelayrecall;
                vals[19] = InputDataValue.orientaion;
                vals[20] = InputDataValue.PhysicalSelfmaintenance;
                vals[21] = InputDataValue.grippingability;
                vals[22] = InputDataValue.word1;
                vals[23] = InputDataValue.word2;
                vals[24] = InputDataValue.word3;
                vals[25] = InputDataValue.wordaverage;
                vals[26] = InputDataValue.worddelayrecall;
                vals[27] = InputDataValue.originalwordrecognition;
                vals[28] = InputDataValue.Newwordrecognize;
                vals[29] = InputDataValue.graphcopy;
                vals[30] = InputDataValue.graphimmediaterecall;
                vals[31] = InputDataValue.graphdelayrecall;
                vals[32] = InputDataValue.lineA;
                vals[33] = InputDataValue.lineB;
                vals[34] = InputDataValue.GDS;
                vals[35] = InputDataValue.CDR;

                if (InputDataValue.strResult == "Normal")
                {
                    vals[36] = data.attribute(36).indexOfValue("Normal");//InputDataValue.Result;
                }
                else if (InputDataValue.strResult == "MCI")
                {
                    vals[36] = data.attribute(36).indexOfValue("MCI");//InputDataValue.Result;
                }
                else
                {
                    vals[36] = data.attribute(36).indexOfValue("AD");//InputDataValue.Result;
                }

                data.add(new Instance(1.0, vals));

                //保存案例库
                ArffSaver saver = new ArffSaver();
                saver.setInstances(data);
                saver.setFile(new java.io.File(".\\InferenceWebService\\Casebase\\Casebase.arff"));
                saver.writeBatch(); 

            }

            data.setClassIndex(data.numAttributes() - 1);

            //2，训练
            Classifier cfs = null;//分类器
            cfs = new weka.classifiers.bayes.BayesNet();
            cfs.buildClassifier(data);
     
            //3，保存训练后的模型

            SaveModel(cfs,"BayesNet.model");   

            return true;
        }

        
        public void SaveModel(Object classifier,String modelname)
        {
            try
            {
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(".\\InferenceWebService\\model\\" + modelname));
                oos.writeObject(classifier);
                oos.flush();
                oos.close();    
            }
            catch(java.io.IOException e)
            {
                e.printStackTrace();
            }

        }

        public Object LoadModel(String file)
        {
            try
            {
                ObjectInputStream ois=new ObjectInputStream(new FileInputStream(file));
                Object classifier=ois.readObject();
                ois.close();
                return classifier;
            }
            catch (java.io.IOException e)
            {
                e.printStackTrace();
                return null;
            }
        
        }


    }

    public class InputData
    {
        public double timeorientation;
        public double placeorientation;
        public double Languageimmediaterecall;
        public double Attentionandcalculation;
        public double shortmemory;
        public double namingobjects;
        public double languageretell;
        public double readingcomprehension;
        public double languageunderstanding;
        public double languageexpression;
        public double drawgraph;
        public double Visualspaceandexecutiveability;
        public double naming;
        public double memory;
        public double attention;
        public double language;
        public double animalnumber;
        public double abstractability;
        public double MoCadelayrecall;
        public double orientaion;
        public double PhysicalSelfmaintenance;
        public double grippingability;
        public double word1;
        public double word2;
        public double word3;
        public double wordaverage;
        public double worddelayrecall;
        public double originalwordrecognition;
        public double Newwordrecognize;
        public double graphcopy;
        public double graphimmediaterecall;
        public double graphdelayrecall;
        public double lineA;
        public double lineB;
//         private int similarity;
//         private int perception;
//         private int CLOX1;
//         private int CLOX2;
        public double GDS;
        public double CDR;
        //public double Result;//0表示正常，1表示MCI，2表示AD

        public string strResult;
    

    }

    public class OutputData
    {
        private string sResult;
    }

}
