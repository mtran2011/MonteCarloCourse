Option Explicit
Option Base 1
' Class Module: CObligor

' Member variable
Public Name As String
Private m_InduCode As Integer
Private m_CtryCode As Integer
Private m_Rating As String
Private m_PDCurve() As Double 'todo

' InduCode property
Public Property Get InduCode() As Integer 
    InduCode = m_InduCode
End Property

Public Property Let InduCode(value As Integer)    
    m_InduCode = value
End Property

' CtryCode property
Public Property Get CtryCode() As Integer 
    CtryCode = m_CtryCode 
End Property

Public Property Let CtryCode(value As Integer)    
    m_CtryCode = value
End Property

' Rating property
Public Property Get CtryCode() As Integer 
    CtryCode = m_CtryCode 
End Property

Public Property Let CtryCode(value As Integer)    
    m_CtryCode = value
End Property


' PDCurve property


' https://excelmacromastery.com/vba-class-modules/#A_Quick_Guide_to_the_VBA_Class_Module
' http://www.cpearson.com/excel/classes.aspx