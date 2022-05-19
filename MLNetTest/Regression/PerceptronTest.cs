using System;
using System.Collections.Generic;
using System.Linq;
using MLNet;
using MLNet.Losses;
using MLNet.Models;
using MLNet.Optimizers;
using MLNet.Utils;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class PerceptronTest : AbstractUnitTest
    {
        private readonly string singledata = @"..\..\..\..\DataSet\iris.data";

        public PerceptronTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            (X, Y, Dict) = np2.loadClassifyData(singledata);
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }
        protected Dictionary<int, string> Dict { set; get; }
        protected int[] Classes => Dict.Select(a => a.Key).ToArray();

        [Fact]
        public void Fit()
        {
            var per = new Perceptron(Classes);

            per.GiveOptimizer(new Adam(1E-1));
            per.GiveLoss(new SoftmaxMutlitClassLoss(Classes.Length));

            per.Fit(X, Y, new TrainPlan(1));
            print(per);
            var res = per.Predict(X);
        }
    }
}