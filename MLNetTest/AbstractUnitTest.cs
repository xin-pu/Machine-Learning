﻿using MathNet.Numerics.Random;
using Xunit.Abstractions;

namespace MLNetTest
{
    public class AbstractUnitTest
    {
        public AbstractUnitTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        internal SystemRandomSource SystemRandomSource => SystemRandomSource.Default;
        internal ITestOutputHelper _testOutputHelper { get; }

        internal void Print<T>(T[] array)
        {
            _testOutputHelper.WriteLine(string.Join(",", array));
        }

        internal void Print(object obj)
        {
            _testOutputHelper.WriteLine(obj.ToString());
        }
    }
}