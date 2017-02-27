# nltk-krippendorff-validation

The goal of this script is to validate the nltk implementation of the Krippendorff agreement coefficient alpha (Kalpha) with an evaluation dataset used in Hayes and Krippendorff, 2007.


**As of 27/02/2017**: investigating why the Kalpha values don't match.
  * Krippendorff's ordinal alpha of the evaluation dataset is reported as `0.7598`. 
  * The implementation of the NLTK agreement module returns a Krippendorff alpha value of `0.2058`. 


**As of 28/02/2017**: the Kalpha [implementation by Thomas Grill] (<https://github.com/grrrr/krippendorff-alpha>) returns the values as reported in the paper.
     
     ```
     Thomas Grill Kalpha nominal metric: 0.477
     Thomas Grill Kalpha interval metric: 0.757
     ```
   

**References**

* NLTK Kalpha implementation:
  <http://www.nltk.org/_modules/nltk/metrics/agreement.html>

* Krippendorff evaluation dataset source:
  Hayes, A.F., Krippendorff, K., 2007. 
  Answering the Call for a Standard Reliability Measure for Coding Data. 
  Communication Methods and Measures 1, 77–89. doi:10.1080/19312450709336664


