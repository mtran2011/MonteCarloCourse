' https://excelmacromastery.com/vba-class-modules/#A_Quick_Guide_to_the_VBA_Class_Module
' http://www.cpearson.com/excel/classes.aspx
' Class Module: CObligor

' Member variable
Public Name As String
Private pInduCode As Integer
Private pCtryCode As Integer
Private pRating As String
Private pPDCurve() As Double 'todo

' Salary property
Public Property Get InduCode() As Integer 
    InduCode = pInduCode 
End Property 
Public Property Let InduCode(Value As Integer)
    If Value > 0 Then
          pSalary = Value 
    Else
        ' appropriate error code here
    End If
End Property