FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY src/RagInABox/RagInABox.csproj src/RagInABox/
RUN dotnet restore src/RagInABox/RagInABox.csproj
COPY . .
RUN dotnet publish src/RagInABox/RagInABox.csproj -c Release -o /app/publish --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080
ENTRYPOINT ["dotnet", "RagInABox.dll"]
