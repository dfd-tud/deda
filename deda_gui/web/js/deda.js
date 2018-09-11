/********** MISC *********/

$("#menu-toggle").click(function(e) {
    e.preventDefault();
    $("#wrapper").toggleClass("toggled");
});

$(document).ready(function(){
    $('#sidebar-wrapper').load('sidebar.html');
});

$('[data-toggle="popover"]').popover({
    trigger: 'focus'
});

//add additional input field
function addField(){

    //CREATE FORM ROW
    var container = $('#input-container');
    var i = container.children().length;
    var row = document.createElement('div');
    row.className = 'form-row';
    row.id = 'forminput'+i;
    row.style.maxWidth = '550px';
    container.append(row);
    var col1 = document.createElement('div');
    var col2 = document.createElement('div');
    col1.className = 'col-md-11';
    col2.className = 'col-md-1';
    row.append(col1);
    row.append(col2);

    // CREATE AND ADD INPUT FIELD
    var input = document.createElement('input');
    input.id = 'file_selector' + i;
    input.className = 'form-control';
    input.style.marginBottom = '3px';
    input.placeholder = 'Additional file or folder path input.';
    col1.append(input);
    // CREATE AND ADD DELETE BUTTON
    var del = document.createElement('a');
    del.href = '#';
    del.id = 'input_delete' + i;
    del.className = 'btn btn-alert';
    del.innerHTML = '<i class="far fa-trash-alt"></i>';
    del.setAttribute('onclick', 'delField('+i+')');
    col2.append(del);
}

//delete additional input field
function delField(number){
    $('#forminput' + number).remove();
}

function generateLoadingIcon(){
    var loading = document.createElement('i');
    loading.className = 'fas fa-spinner fa-pulse';
    loading.id = 'loading';
    loading.style.marginLeft = '10px';
    return loading;
}

function generateAlertInfo(info){
    var div = document.createElement('div');
    div.className = 'alert alert-danger';
    div.id = 'file_info';
    div.style.marginTop = '10px';
    div.setAttribute('role', "alert");
    div.innerHTML = info;
    return div;
}

function generateSuccessInfo(info){
    var div = document.createElement('div');
    div.className = 'alert alert-success';
    div.id = 'file_info';
    div.style.marginTop = '10px';
    div.setAttribute('role', "alert");
    div.innerHTML = info;
    return div;
}

/********** FORENSIC *********/
async function getForensicFiles() {
    //clear latest results
    sessionStorage.clear();
    $('#resultscard').remove();
    $('#file_info').remove();

    //create loading icon
    var loading = generateLoadingIcon();
    $('#headerform').append(loading);

    //get all input files and folders
    let files = [];
    files.push($('#file_selector').val());
    var size = $('#input-container').children().length;
    for(i=0; i<size;i++){
        var id = '#file_selector'+i;
        files.push($(id).val());
    }

    // Send to Python
    let info = await eel.forensic(files)();
    if(info != ''){
        file_info = generateAlertInfo(info);
        $('#contentheader').after(file_info);
    }
    //remove loading button
    $('#loading').remove();
}

//PRINT FORENSIC RESULTS
eel.expose(printForensicResult);
function printForensicResult(allresults, errors){

    //create result card
    var card = document.createElement('card');
    card.id = 'resultscard';
    card.className = 'card';
    card.style.marginTop = '10px';
    card.style.maxWidth = '500px';
    var cardbody = document.createElement('card-body');
    cardbody.className = 'card-body';

    //Header
    var cardrow = document.createElement('div');
    cardrow.className = 'row';
    var col1 = document.createElement('div');
    var col2 = document.createElement('div');
    var col3 = document.createElement('div');
    var col4 = document.createElement('div');
    col1.className = 'col-md-7';
    col2.className = 'col-md-2';
    col3.className = 'col-md-2';
    col4.className = 'col-md-1';
    var header = document.createElement('h5');
    header.innerHTML = 'Results';
    var btn1 = document.createElement('button');
    var btn2 = document.createElement('button');
    btn1.className = 'btn btn-light btn-sm';
    btn2.className = 'btn btn-light btn-sm';
    btn1.setAttribute('type', "button");
    btn2.setAttribute('type', "button");
    btn1.setAttribute('onclick', 'downloadForensicPdf()');
    btn2.setAttribute('onclick', 'downloadForensicPng()');
    btn1.id = 'downloadforensic_pdf';
    btn2.id = 'downloadforensic_png';
    btn1.innerHTML = '<i class="fas fa-download"></i>PDF'
    btn2.innerHTML = '<i class="fas fa-download"></i>PNG'
    col1.append(header);
    col2.append(btn1);
    col3.append(btn2);
    cardrow.append(col1,col2,col3,col4);
    cardbody.append(cardrow);

    //Result carousel
    var carousel = document.createElement('div');
    carousel.className = 'carousel slide';
    carousel.id = 'res-carousel';
    carousel.setAttribute('data-ride', "carousel");
    carousel.setAttribute('data-interval', "false");

    //carousel controls if results > 1
    if(allresults.length > 1 || ((errors[0].length>0 || errors[1].length>0) && allresults.length>0)){
        var left = document.createElement('a');
        left.className = 'left carousel-control';
        left.id = 'leftcontrol';
        left.href = '#res-carousel';
        left.setAttribute('data-slide', "prev");
        left.innerHTML = '<i class="fas fa-chevron-circle-left"></i>'
        var right = document.createElement('a');
        right.className = 'right carousel-control';
        right.id = 'rightcontrol';
        right.style.marginLeft = '3px';
        right.href = '#res-carousel';
        right.setAttribute('data-slide', "next");
        right.innerHTML = '<i class="fas fa-chevron-circle-right"></i>'
        carousel.append(left, right);
    }

    //carousel inner
    var carouselinner = document.createElement('div');
    carouselinner.id = 'carouselinner';
    carouselinner.className = 'carousel-inner'

    //save results for pdf creation
    var pdftables = [];
    var pdftdms = [];

    //iterate through all results
    for(var i = 0; i<allresults.length; i++){

        //create new carousel items
        var carouselitem = document.createElement('div');
        //carousel item
        if(i==0){carouselitem.className= 'carousel-item active';}
        else{carouselitem.className = 'carousel-item';}

        //information table
        dec_table = document.createElement('table');
        dec_table.className = 'table table-bordered table-sm table-hover';
        dec_table.id = 'resulttable';
        decbody = document.createElement('tbody');

        var pdfrows = [];

        for(var x=0; x<allresults[i][1].length; x++){
            var tr = document.createElement('tr');
            var th = document.createElement('th');
            var td = document.createElement('td');
            td.innerHTML = allresults[i][1][x];
            if(x==0){
                th.innerHTML = 'Analysed Image';
                var pdfrow = ['Analysed Image', allresults[i][1][x]];
            }
            if(x==1){
                th.innerHTML = 'Detected Pattern';
                var pdfrow = ['Detected Pattern', allresults[i][1][x]];
            }
            if(x==2){
                th.innerHTML = 'Manufacturer';
                var pdfrow = ['Manufacturer', allresults[i][1][x]];
            }
            if(x==3){
                th.innerHTML = 'Serial Number';
                var pdfrow = ['Serial Number', allresults[i][1][x]];
            }
            if(x==4){
                th.innerHTML = 'Timestamp';
                var pdfrow = ['Timestamp', allresults[i][1][x]];
            }
            if(x==5){
                th.innerHTML = 'Raw';
                var pdfrow = ['Raw', allresults[i][1][x]];
            }
            if(x==6){
                th.innerHTML = 'Dot Count per Pattern';
                var pdfrow = ['Dot Count per Pattern', allresults[i][1][x]];
            }
            tr.append(th, td);
            pdfrows.push(pdfrow);
            decbody.append(tr);
        }
        pdftables.push(pdfrows);
        dec_table.append(decbody);

        //Tracking Dot Matrix
        result_matrix = document.createElement('table');
        result_matrix.className = 'table table-bordered table-sm table-dark table-hover text-center';
        result_matrix.id = 'tdmtable';
        matrixbody = document.createElement('tbody');

        var pdfrows = [];
        var tdmlist = allresults[i][0];

        for(var j=0; j<tdmlist.length;j++){
            if(tdmlist[j].length>0){
                var pdfrow = []
                //Header
                var tr = document.createElement('tr');
                var th = document.createElement('th');
                th.setAttribute('scope', "row");
                th.innerHTML = tdmlist[j].charAt(0);
                tr.append(th);
                pdfrow.push(tdmlist[j].charAt(0));
                //Tracking dot Matrice
                for(var k=2; k<tdmlist[j].length; k=k+2){
                    var td = document.createElement('td');
                    if(tdmlist[j].charAt(k) === '.'){
                        td.style.color = 'yellow';
                        td.innerHTML = '<i class="far fa-dot-circle"></i>';
                        pdfrow.push('x');
                    }
                    else{
                        td.innerHTML = tdmlist[j].charAt(k);
                        pdfrow.push(tdmlist[j].charAt(k));
                    }
                    tr.append(td);
                }
                matrixbody.append(tr);
                pdfrows.push(pdfrow)
            }
        }
        result_matrix.append(matrixbody);
        pdftdms.push(pdfrows);

        carouselitem.append(document.createElement('br'));
        carouselitem.appendChild(dec_table);
        carouselitem.appendChild(document.createElement('br'));
        carouselitem.append(result_matrix);
        carouselinner.append(carouselitem);
    }
    sessionStorage.setItem('pdftables', JSON.stringify(pdftables));
    sessionStorage.setItem('pdftdms', JSON.stringify(pdftdms));

    //Errors
    if(errors[0].length+errors[1].length>1 || ((errors[0].length>0 || errors[1].length>0) && allresults.length>0)){
        //create error carousel item
        var carouselitem = document.createElement('div');
        if(allresults.length==0){carouselitem.className= 'carousel-item active';}
        else{carouselitem.className = 'carousel-item';}
        var u = document.createElement('h6');
        var table = document.createElement('table');
        table.className = 'table table-bordered table-sm';
        table.id = 'resulttable';
        var tbody = document.createElement('tbody');
        var tr = document.createElement('tr');
        var th = document.createElement('th');
        th.setAttribute('scope', "row");
        var td = document.createElement('td')

        //full failures
        if(errors[0].length>0){
            u.innerHTML = 'No tracking dot pattern detected. For best results try a 300 dpi scan and a lossless file format.';
            th.innerHTML = 'Files';
            var ferror = '';
            //paths
            for(var i=0; i<errors[0].length; i++){
                var path = errors[0][i];
                if(i==0 || errors[0].length==1){
                    ferror = path;
                }
                else{
                    ferror += '<hr id="table_hr">' + path;
                }
            }
            td.innerHTML = ferror;
            tr.append(th, td);
            tbody.append(tr);
            table.append(tbody);
            carouselitem.append(document.createElement('br'),u,table);
        }
        //Extract Failures
        if(errors[1].length>0){
            u.innerHTML = 'Tracking Dots detected, but no valid TDM could be extracted. Try $ deda_extract_yd INPUTFILE for more information.';
            tr.innerHTML = 'Files';
            var ferror = '';
            for(var i=0; i<errors[1].length; i++){
                var path = errors[1][i];
                if(i==0 || errors[1].length==1){
                    ferror = path;
                }
                else{
                    ferror += '<hr id="table_hr">' + path;
                }
            }
            td.innerHTML = ferror;
            tr.append(th,td);
            tbody.append(tr);
            table.append(tbody);
            carouselitem.append(document.createElement('br'),u,table);
        }
        //add item
        carouselinner.append(carouselitem);
    }
    carousel.append(carouselinner);
    cardbody.append(carousel);
    card.append(cardbody);
    $('#content').append(card);
}

// DOWNLOAD FORENSIC RESULT AS PNG
function downloadForensicPng() {
    var node = document.getElementsByClassName('active');
    domtoimage.toPng(node[0], {bgcolor:'#efefef'})
        .then(function (dataUrl) {
            var link = document.createElement("a");
            document.body.appendChild(link);
            link.download = 'trackingdots.png';
            link.href = dataUrl;
            link.click();
        })
        .catch(function (error) {
            console.error('Something went wrong!', error);
        });
}

// DOWNLOAD FORENSIC RESULTS AS PDF
function downloadForensicPdf() {
    var resultrows = JSON.parse(sessionStorage.getItem('pdftables'));
    var resultcolumns = ['Attribute', 'Result'];
    var tdmrows = JSON.parse(sessionStorage.getItem('pdftdms'));
    var doc = new jsPDF('p', 'pt');
    doc.setFontSize(22);
    doc.text('Results of Tracking Dot Analysis', 40, 50);

    if(resultrows.length==tdmrows.length){

        for(var i=0; i<resultrows.length;i++){
            var tdmcolumns = [];
            doc.autoTable(resultcolumns, resultrows[i], {
                headerStyles: {
                    fillColor: [43, 62, 80],
                    textColor: 235,
                },
                tableWidth: 'wrap',
                styles: {fontSize: 10},
                alternateRowStyles: {
                    fillColor: [222, 226, 230]
                },
                startY: 100,
            });

            let first = doc.autoTable.previous;

            for(var k = 0;k<tdmrows[i][0].length;k++){
                tdmcolumns.push('');
            }
            doc.autoTable(tdmcolumns, tdmrows[i], {
                showHeader: 'never',
                startY: first.finalY + 20,
                tableWidth: 'wrap',
                tableLineColor: [222, 226, 230],
                tableLineWidth: 0.7,
                styles: {cellPadding: 3.0, fontSize: 8},

                bodyStyles: {
                    fillColor: [33, 37, 41],
                    textColor: [255,255,255]
                },
                margin: {top: 60},
                alternateRowStyles: {
                    fillColor: [33, 37, 41],
                    textColor: [255,255,255]
                },
            });
            if(i<resultrows.length-1){doc.addPage();}
        }
        doc.save('resultForensic.pdf');
    }
}


/********** COMPARE *********/
async function getCompareFiles() {
    //clear latest results
    sessionStorage.clear();
    $('#resultscard').remove();
    $('#file_info').remove();

    //create loading icon
    var loading = generateLoadingIcon();
    $('#headerform').append(loading);

    //get all input files and folders
    let files = [];
    files.push($('#file_selector').val());
    var size = $('#input-container').children().length;
    for(i=0; i<size;i++){
        var id = '#file_selector'+i;
        files.push($(id).val());
    }

    // Send to Python
    let info = await eel.compare(files)();
    if(info != ''){
        var file_info = generateAlertInfo(info);
        $('#contentheader').after(file_info);
    }
    //remove loading button
    $('#loading').remove();
}

//PRINT COMPARE RESULTS
eel.expose(printCompareResult);
function printCompareResult(info, result_compare){

    //Create info
    var file_info = generateSuccessInfo(info);
    $('#contentheader').after(file_info);

    //create result card
    var card = document.createElement('card');
    card.id = 'resultscard';
    card.className = 'card';
    card.style.marginTop = '10px';
    var cardbody = document.createElement('card-body');
    cardbody.className = 'card-body';

    //card header
    var cardrow = document.createElement('div');
    cardrow.className = 'row';
    var col1 = document.createElement('div');
    var col2 = document.createElement('div');
    var col3 = document.createElement('div');
    col1.className = 'col-md-9';
    col2.className = 'col-md-2';
    col3.className = 'col-md-1';
    var header = document.createElement('h5');
    header.innerHTML = 'Results';
    var btn = document.createElement('button');
    btn.className = 'btn btn-light btn-sm';
    btn.setAttribute('type', "button");
    btn.setAttribute('onclick', 'downloadComparePdf()');
    btn.id = 'downloadcompare_pdf';
    btn.innerHTML = '<i class="fas fa-download"></i>PDF'
    col1.append(header);
    col2.append(btn);
    cardrow.append(col1,col2,col3);
    cardbody.append(cardrow);

    //create result table
    var resulttable = document.createElement('table');
    resulttable.className = 'table table-bordered table-sm table-hover';
    resulttable.id = 'resulttable';

    //head
    var thead = document.createElement('thead');
    var tr = document.createElement('tr');
    var th1 = document.createElement('th');
    th1.setAttribute('scope', "col");
    th1.innerHTML = '#';
    var th2 = document.createElement('th');
    th2.setAttribute('scope', "col");
    th2.innerHTML = 'Manufacturer';
    var th3 = document.createElement('th');
    th3.setAttribute('scope', "col");
    th3.innerHTML = 'Files';
    tr.append(th1,th2,th3);
    thead.append(tr);
    resulttable.append(thead);

    var pdfcolumns = ['#', 'Manufacturer', 'Files'];
    var pdfrows = [];
    //content
    var tbody = document.createElement('tbody');
    for(var i=0;i<result_compare.length;i++){
        pdfrow = [];
        var tr = document.createElement('tr');
        var th = document.createElement('th');
        th.setAttribute('scope', "row");
        th.innerHTML = result_compare[i][0];
        pdfrow.push(result_compare[i][0]);
        var td1 = document.createElement('td');
        td1.innerHTML = result_compare[i][1];
        pdfrow.push(result_compare[i][1]);
        var td2 = document.createElement('td');
        f = '';
        fpdf = '';
        //paths
        for(var j=0;j<result_compare[i][2].length;j++){
            var path = result_compare[i][2][j];

            if(j==0 || result_compare[i][2].length==1){
                f = path;
                var fname = path.split('/');
                fpdf = fname[fname.length-1];
            }
            else{
                f += '<hr id="table_hr">' + path;
                var fname = path.split('/');
                fpdf += '\n' + fname[fname.length-1];
            }
        }
        td2.innerHTML = f;
        tr.append(th,td1,td2);
        tbody.append(tr);
        pdfrow.push(fpdf);
        pdfrows.push(pdfrow);
    }
    resulttable.append(tbody);
    cardbody.append(document.createElement('br'));
    cardbody.append(resulttable);
    card.append(cardbody);
    $('#content').append(card);
    resultpdf = [pdfcolumns, pdfrows];
    sessionStorage.setItem('resultpdf', JSON.stringify(resultpdf));
}

// DOWNLOAD COMPARE RESULTS AS PDF
function downloadComparePdf() {
    var resultpdf = JSON.parse(sessionStorage.getItem('resultpdf'));
    var resultcolumns = resultpdf[0];
    var resultrows = resultpdf[1]

    var doc = new jsPDF('p', 'pt');
    doc.setFontSize(22);
    doc.text('Results of Compare Analysis', 40, 50);

    doc.autoTable(resultcolumns, resultrows, {
        startY: 100,
        tableWidth: 'wrap',
        styles: {fontSize: 9},
        headerStyles: {
            fillColor: [43, 62, 80],
        },
        alternateRowStyles: {
            fillColor: [222, 226, 230]
        },
    });
    doc.save('resultCompare.pdf');
}

/************ SCAN ANON ***************/
async function anonymizeScan() {
    //reset old infos
    $('#resulttable').remove();
    $('#file_info').remove();

    //create loading icon
    var loading = generateLoadingIcon();
    $('#headerform').append(loading);

    //get input file
    let file = $('#file_selector').val();

    // Send to Python
    let info = await eel.anonScanAction(file)();
    if(info != ''){
        var file_info = generateAlertInfo(info);
        $('#contentheader').after(file_info);
    }
    $('#loading').remove();
}

//PRINT ANON SCAN RESULTS
eel.expose(printAnonScan);
function printAnonScan(info, result) {
    //create success info
    var file_info = generateSuccessInfo(info);
    $('#contentheader').after(file_info);

    var t = document.createElement('table');
    t.className = 'table table-bordered table-sm table-dark table-hover text-center';
    t.style.maxWidth = '600px';
    t.id = 'resulttable';
    t.marginTop = '10px';
    var tb = document.createElement('tbody');
    for(var i=0;i<3;i++){
        var tr = document.createElement('tr');
        var th = document.createElement('th');
        th.setAttribute('scope', "row");
        if(i==0){th.innerHTML = 'Path';}
        if(i==1){th.innerHTML = 'Image';}
        if(i==2){th.innerHTML = 'Anonymized Image';}
        var td = document.createElement('td');
        td.innerHTML = result[i];
        tr.append(th, td);
        tb.append(tr);
    }
    t.append(tb);
    $('#content').append(t);
}



/************ PRINT ANON ***************/
async function generateAnonMask() {
    $('#file_info').remove();

    //create loading icon
    var loading = generateLoadingIcon();
    $('#loading_div').append(loading);

    //get scanned calibration file
    let calibrationfile = $('#file_selector').val();

    // Send to Python
    let info = await eel.generateMask(calibrationfile)();
    done = info.split(".")[0];
    if(done == 'Done'){
        var file_info = generateSuccessInfo(info);
        $('#step2').after(file_info);
    }
    else if(info != ''){
        var file_info = generateAlertInfo(info);
        $('#step2').after(file_info);
    }
    $('#loading').remove();
}


async function generateAnonPrintFile(){
    $('#resulttable').remove();
    $('#file_info').remove();

    //create loading icon
    var loading = generateLoadingIcon();
    $('#loading_div').append(loading);

    let page = $('#file_selector').val();
    let mask = $('#mask_selector').val();

    let x = $('#x').val();
    let y = $('#y').val();
    let dotradius = $('#dotradius').val();

    // Send to Python
    let info = await eel.applyMask(page, mask, x, y, dotradius)();
    if(info != ''){
        var file_info = generateAlertInfo(info);
        $('#contentheader').after(file_info);
    }
    $('#loading').remove();
}


//PRINT ANON MASK RESULT
eel.expose(printAnonMaskResult);
function printAnonMaskResult(info, result) {
    var file_info = generateSuccessInfo(info);
    $('#contentheader').after(file_info);

    var t = document.createElement('table');
    t.className = 'table table-bordered table-sm table-dark table-hover text-center';
    t.style.maxWidth = '600px';
    t.id = 'resulttable';
    t.marginTop = '10px';
    var tb = document.createElement('tbody');
    for(var i=0;i<5;i++){
        var tr = document.createElement('tr');
        var th = document.createElement('th');
        th.setAttribute('scope', "row");
        if(i==0){th.innerHTML = 'Masked Document';}
        if(i==1){th.innerHTML = 'Info File';}
        if(i==2){th.innerHTML = 'x-Offset';}
        if(i==3){th.innerHTML = 'y-Offset';}
        if(i==4){th.innerHTML = 'Dotradius';}
        var td = document.createElement('td');
        td.innerHTML = result[i];
        tr.append(th, td);
        tb.append(tr);
    }
    t.append(tb);
    $('#content').append(t);
}


//GENERATE OWN PATTERN
async function generatePattern(){
    $('#resultscard').remove();
    $('#file_info').remove();

    //create loading icon
    var loading = generateLoadingIcon();
    $('#loading_div').append(loading);

    let pdf = $('#file_selector').val();
    let datetime = $('#datetime').val();
    let serial = $('#serial').val();
    let manufacturer = $('#manufacturer').val();
    let dotradius = $('#dotradius').val();

    // Send to Python
    let info = await eel.generatePattern(pdf, datetime, serial, manufacturer, dotradius)();
    done = info.split(".")[0];
    if(info != ''){
        var file_info = generateAlertInfo(info);
        $('#contentheader').after(file_info);
    }
    $('#loading').remove();
}

//PRINT CREATE PATTERN RESULT
eel.expose(printCreateResult);
function printCreateResult(tdm, results){
    //create result card
    var card = document.createElement('card');
    card.id = 'resultscard';
    card.className = 'card';
    card.style.marginTop = '10px';
    card.style.maxWidth = '500px';
    var cardbody = document.createElement('card-body');
    cardbody.className = 'card-body';

    //Header
    var cardrow = document.createElement('div');
    cardrow.className = 'row';
    var col1 = document.createElement('div');
    var col2 = document.createElement('div');
    var col3 = document.createElement('div');
    col1.className = 'col-md-9';
    col2.className = 'col-md-2';
    col3.className = 'col-md-1';
    var header = document.createElement('h5');
    header.innerHTML = 'Results';
    var btn = document.createElement('button');
    btn.className = 'btn btn-light btn-sm';
    btn.setAttribute('type', "button");
    btn.setAttribute('type', "button");
    btn.setAttribute('onclick', 'downloadCreatePng()');
    btn.id = 'downloadcreate_pdf';
    btn.innerHTML = '<i class="fas fa-download"></i>PNG'
    col1.append(header);
    col2.append(btn);
    cardrow.append(col1,col2,col3);
    cardbody.append(cardrow);

    //Information table
    dec_table = document.createElement('table');
    dec_table.className = 'table table-bordered table-sm table-hover';
    dec_table.id = 'resulttable';
    dec_table.style.marginTop = '20px';
    decbody = document.createElement('tbody');

    for(var i=0; i<results.length; i++){
        var tr = document.createElement('tr');
        var th = document.createElement('th');
        var td = document.createElement('td');
        td.innerHTML = results[i];
        if(i==0){th.innerHTML = 'Path';}
        if(i==1){th.innerHTML = 'Pattern';}
        if(i==2){th.innerHTML = 'Manufacturer';}
        if(i==3){th.innerHTML = 'Serial';}
        if(i==4){th.innerHTML = 'Timestamp';}
        if(i==5){th.innerHTML = 'Dotradius';}
        if(i==6){th.innerHTML = 'Dot count';}
        tr.append(th, td);
        decbody.append(tr);
    }
    dec_table.append(decbody);

    //Tracking Dot Matrix
    result_matrix = document.createElement('table');
    result_matrix.className = 'table table-bordered table-sm table-dark table-hover text-center';
    result_matrix.id = 'tdmtable';
    matrixbody = document.createElement('tbody');

    for(var j=0; j<tdm.length;j++){
        if(tdm[j].length>0){
            //Header
            var tr = document.createElement('tr');
            var th = document.createElement('th');
            th.setAttribute('scope', "row");
            th.innerHTML = tdm[j].charAt(0);
            tr.append(th);
            //Tracking dot Matrice
            for(var k=2; k<tdm[j].length; k=k+2){
                var td = document.createElement('td');
                if(tdm[j].charAt(k) === '.'){
                    td.style.color = 'yellow';
                    td.innerHTML = '<i class="far fa-dot-circle"></i>';
                }
                else{
                    td.innerHTML = tdm[j].charAt(k);
                }
                tr.append(td);
            }
            matrixbody.append(tr);
        }
    }
    result_matrix.append(matrixbody);
    cardbody.append(dec_table, result_matrix);
    card.append(cardbody);
    $('#content').append(card);
}

// DOWNLOAD CREATE RESULT AS PNG
function downloadCreatePng() {
    res = document.getElementById('resultscard');
    domtoimage.toPng(res, {bgcolor:'#efefef'})
        .then(function (dataUrl) {
            var link = document.createElement("a");
            document.body.appendChild(link);
            link.download = 'trackingdots.png';
            link.href = dataUrl;
            link.click();
        })
        .catch(function (error) {
            console.error('Something went wrong!', error);
        });
}
