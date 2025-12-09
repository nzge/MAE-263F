# Homework 5


Run Homework5.ipynb from the start to run the full simulation.
There are some code lines commented out that are exclusive to running on google colab.


## Mechanics of Flexible Structures & Soft Robots
### MAE-263F

```
Fall 2025
```


```
Due Date: 11:59 PM, 12/05/
```
```
You should create a single GitHub repository for this class and share it with the instructor
(khalidjm@seas.ucla.edu). All the homeworks, reports, presentations, and proposal should be
uploaded to this repository. Do not create a separate repository for each assignment. Within your
repository, create a separate folder for each assignment (e.g.,homework_1,homework_2,homework_3,
homework_4,homework_5,proposal,midterm_report, andfinal_report).
```
```
Submission Instructions:Your submission on BruinLearn should only contain the URL to your
GitHub repository. Your GitHub repository should include the following items:
```
1. Report in PDF format: Submit a report in .pdf format (file name should be
    Homework5_LASTNAME.pdf, replacingLASTNAMEwith your last name) addressing the questions
    asked in the deliverables. Include all the plots and figures requested in the assignment and
    discuss them in the report. See the syllabus for formatting requirements. As stated in the
    syllabus, you must use one of the provided templates.
2. Source code: The submission should include one main file named exactly as
    HomeworkX.[ext](whereXis the homework number) along with any additional files (e.g.,
    functions or text files) as necessary. The main file should require no more than a single com-
    mand to run or one click for execution.
3. README file:Add aREADME.mdfile on GitHub to provide clear instructions on how to run
    your code and describe the purpose of each file included in your repository.

# Deformation of a Clamped Thin Beam Using a Plate Model

In this homework, you will simulate the static deformation of a thin, rectangular beam under its
own weight using thediscrete plate modeland compare your numerical results against the classical
Euler-Bernoulli beam theory prediction.

Geometry & Material.
Beam dimensions:
lengthl= 0.1 m,
widthw= 0.01 m,
thicknessh= 0.002 m.

Material parameters:
Young’s modulusY= 10^7 Pa,
densityρ= 1000 kg/m^3.

Section properties (for Euler–Bernoulli comparison).
Cross-sectional area:A=wh.


Homework 1 2 of 3

```
(- 0.0125,0) (0,0) (0.0125,0) (0.025,0) (0.0375,0) (0.05,0) (0.0625,0) (0.075,0) (0.0875,0) (0.1,0)
```
```
(- 0.0125,0.01) (0,0.01) (0.0125,0.01)(0.025,0.01)(0.0375,0.01) (0.05,0.01) (0.0625,0.01)(0.075,0.01) (0.0875,0.01) (0.1,0.01)
```
```
Fixed node
x Free node
```
```
y
```
```
z
```
Second moment of area:I=

```
wh^3
12
```
### .

Distributed load from gravity:q=ρAg.

Mesh.
Use the mesh shown in the figure above. The length of the free portion of the plate isl= 0.1 m.
The four leftmost nodes are fixed to enforce the clamped boundary condition. Gravity acts in the
negativez-direction.

Boundary Condition.
The left edge of the plate (atx= 0) is fully clamped: all displacement and rotation components are
fixed.

Plate Model Simulation.
Model the domainl×was a thin plate of thicknessh. Compute the static deformation by solving
the equilibrium equations until the configuration reaches steady state.

Let the tip displacement at the centerline of the free edgex=lbe

```
δplate(t) =ztip(t)−ztip(t= 0).
```
Report the steady displacement in your report and include a plot of the tip displacement as a
function of time.

Comparison with Euler–Bernoulli Beam Theory.
For a cantilever beam under uniform loadq(Newton per meter), the Euler-Bernoulli tip displacement
is

```
δEB=
q l^4
8 Y I
```
### .

In your report, compare steady displacement from discrete plate simulation with the prediction from
Euler-Bernoulli beam theory.

Deliverables.At minimum, your report must include:

- Steady displacement from discrete plate simulation (δplate) and theoretical prediction (δEB),
    and their normalized difference.
- A plot ofδplatevs. timet.

Copyright notice: Do not copy, distribute, republish, upload, post or transmit anything from the files hosted on the
class website without written permission of the instructor


Homework 1 3 of 3

Figure Requirements. All plots must include labeled axes with physicalunits, and each figure
must have a caption with a figure number. Export high-resolution vector graphics (PDF). Do not
use screenshots.

Copyright notice: Do not copy, distribute, republish, upload, post or transmit anything from the files hosted on the
class website without written permission of the instructor


